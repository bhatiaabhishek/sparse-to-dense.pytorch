import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
import drn
from torchvision.models import resnet
from erf_blocks import DownsamplerBlock, UpsamplerBlock, non_bottleneck_1d

BatchNorm = nn.BatchNorm2d
class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride).cuda()) # currently not compatible with running on CPU
        self.weights[:,:,0,0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)

def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class Decoder(nn.Module):
    # Decoder is the base class for all decoders

    names = ['deconv2', 'deconv3', 'upconv', 'upproj']

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class DeConv(Decoder):
    def __init__(self, in_channels, kernel_size):
        assert kernel_size>=2, "kernel_size out of range: {}".format(kernel_size)
        super(DeConv, self).__init__()

        def convt(in_channels):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                  (module_name, nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size,
                        stride,padding,output_padding,bias=False)),
                  ('batchnorm', nn.BatchNorm2d(in_channels//2)),
                  ('relu',      nn.ReLU(inplace=True)),
                ]))

        self.layer1 = convt(in_channels)
        self.layer2 = convt(in_channels // 2)
        self.layer3 = convt(in_channels // (2 ** 2))
        self.layer4 = convt(in_channels // (2 ** 3))

class UpConv(Decoder):
    # UpConv decoder consists of 4 upconv modules with decreasing number of channels and increasing feature map size
    def upconv_module(self, in_channels):
        # UpConv module: unpool -> 5*5 conv -> batchnorm -> ReLU
        upconv = nn.Sequential(collections.OrderedDict([
          ('unpool',    Unpool(in_channels)),
          ('conv',      nn.Conv2d(in_channels,in_channels//2,kernel_size=5,stride=1,padding=2,bias=False)),
          ('batchnorm', nn.BatchNorm2d(in_channels//2)),
          ('relu',      nn.ReLU()),
        ]))
        return upconv

    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.layer1 = self.upconv_module(in_channels)
        self.layer2 = self.upconv_module(in_channels//2)
        self.layer3 = self.upconv_module(in_channels//4)
        self.layer4 = self.upconv_module(in_channels//8)

class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()
        self.conv1 = nn.Conv2d(1,16,5)
        self.conv_img = nn.Conv2d(3,16,5)
        self.bn1 = BatchNorm(16)
        self.bn_img = BatchNorm(16)
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = model #nn.Sequential(*list(model.children())[:-2])
        self.aspp = drn.aspp([[512,512,1],
                              [512,512,6],
                              [512,512,12],
                              [512,512,18]])
        self.seg = nn.Conv2d(512, classes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        ## [1x1, 32] for channel reduction.
        #self.conv2 = nn.Conv2d(64, 32, 1, bias=False)
        #self.bn2 = nn.BatchNorm2d(32)
        #self.relu2 = nn.ReLU()

        self.last_conv = nn.Sequential(
                                       nn.Conv2d(704, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU())
        for m in self.last_conv.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,math.sqrt(2. / n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 8, stride=4, padding=3,
                                    output_padding=0, groups=classes,
                                    bias=False)
            #fill_up_weights(up)
            #up.weight.requires_grad = False
            self.up = up
        self.up_4 = nn.ConvTranspose2d(512, 512, 8, stride=2, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
        m = self.up
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m = self.up_4
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    def forward(self,x_inp):
        x_input = x_inp['d']
        x_rgb = x_inp['rgb']
        x = F.relu(self.bn1(self.conv1(x_input)))
        img = F.relu(self.bn_img(self.conv_img(x_rgb)))
        concat_feat = torch.cat((x,img),dim=1)
        base_x, low_level_feats  = self.base(concat_feat)
        x = self.aspp(base_x)
        #print(low_level_feat.size())
        x = torch.cat((x,low_level_feats[3]),dim=1)
        x = F.interpolate(x, size=(int(math.ceil(-1+x_input.size()[-2]/4)),
                                            int(math.ceil(-1+x_input.size()[-1]/4))), mode='bilinear', align_corners=True)
        #print("bfor up = ", x.shape)
        #x = self.up_4(x) 
           
        #llf = self.conv2(low_level_feat)
        #llf = self.bn2(llf)
        #llf = self.relu2(llf)
        x = torch.cat((x,low_level_feats[2]),dim=1)
        x = self.last_conv(x)
        x = self.seg(x)
        #y = self.up(x)
        y = nn.UpsamplingBilinear2d(size=(x_input.shape[2],x_input.shape[3]))(x)
        #y = F.interpolate(x, size=x_input.size()[2:], mode='bilinear', align_corners=True)

        return y


#class DRNSeg(nn.Module):
#    def __init__(self, model_name, classes, pretrained_model=None,
#                 pretrained=True, use_torch_up=False):
#        super(DRNSeg, self).__init__()
#        self.conv1 = nn.Conv2d(1,16,5)
#        self.conv_img = nn.Conv2d(3,16,5)
#        self.bn1 = BatchNorm(16)
#        self.bn_img = BatchNorm(16)
#        model = drn.__dict__.get(model_name)(
#            pretrained=pretrained, num_classes=1)
#        pmodel = nn.DataParallel(model)
#        if pretrained_model is not None:
#            pmodel.load_state_dict(pretrained_model)
#        self.base = model #nn.Sequential(*list(model.children())[:-2])
#        self.aspp = drn.aspp([[512,512,1],
#                              [512,512,6],
#                              [512,512,12],
#                              [512,512,18]])
#        self.seg = nn.Conv2d(512, classes,
#                             kernel_size=1, bias=True)
#        self.softmax = nn.LogSoftmax(dim=1)
#        m = self.seg
#        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#        m.weight.data.normal_(0, math.sqrt(2. / n))
#        m.bias.data.zero_()
#        ## [1x1, 32] for channel reduction.
#        #self.conv2 = nn.Conv2d(64, 32, 1, bias=False)
#        #self.bn2 = nn.BatchNorm2d(32)
#        #self.relu2 = nn.ReLU()
#
#        self.last_conv = nn.Sequential(
#                                       nn.Conv2d(576, 512, kernel_size=3, stride=1, padding=1, bias=False),
#                                       nn.BatchNorm2d(512),
#                                       nn.ReLU())
#        for m in self.last_conv.modules():
#            if isinstance(m,nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                m.weight.data.normal_(0,math.sqrt(2. / n))
#            elif isinstance(m,nn.BatchNorm2d):
#                m.weight.data.fill_(1)
#                m.bias.data.zero_()
#        if use_torch_up:
#            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
#        else:
#            up = nn.ConvTranspose2d(classes, classes, 8, stride=4, padding=3,
#                                    output_padding=0, groups=classes,
#                                    bias=False)
#            #fill_up_weights(up)
#            #up.weight.requires_grad = False
#            self.up = up
#        self.up_4 = nn.ConvTranspose2d(512, 512, 8, stride=2, padding=4,
#                                    output_padding=0, groups=classes,
#                                    bias=False)
#        m = self.up
#        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#        m.weight.data.normal_(0, math.sqrt(2. / n))
#        m = self.up_4
#        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#        m.weight.data.normal_(0, math.sqrt(2. / n))
#    def forward(self,x_inp):
#        x_input = x_inp['d']
#        x_rgb = x_inp['rgb']
#        x = F.relu(self.bn1(self.conv1(x_input)))
#        img = F.relu(self.bn_img(self.conv_img(x_rgb)))
#        concat_feat = torch.cat((x,img),dim=1)
#        base_x, low_level_feat  = self.base(concat_feat)
#        x = self.aspp(base_x)
#        #print(x.size())
#        #print(low_level_feat.size())
#        x = F.interpolate(x, size=(int(math.ceil(-1+x_input.size()[-2]/4)),
#                                            int(math.ceil(-1+x_input.size()[-1]/4))), mode='bilinear', align_corners=True)
#        #print("bfor up = ", x.shape)
#        #x = self.up_4(x) 
#           
#        #llf = self.conv2(low_level_feat)
#        #llf = self.bn2(llf)
#        #llf = self.relu2(llf)
#        x = torch.cat((x,low_level_feat),dim=1)
#        x = self.last_conv(x)
#        x = self.seg(x)
#        #y = self.up(x)
#        y = nn.UpsamplingBilinear2d(size=(x_input.shape[2],x_input.shape[3]))(x)
#        #y = F.interpolate(x, size=x_input.size()[2:], mode='bilinear', align_corners=True)
#
#        return y
#
#    def optim_parameters(self, memo=None):
#        for param in self.base.parameters():
#            yield param
#        for param in self.aspp.parameters():
#            yield param
#        for param in self.seg.parameters():
#            yield param
#        for param in self.last_conv.parameters():
#            yield param
#        for param in self.up.parameters():
#            yield param

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,5)
        self.conv_img = nn.Conv2d(3,16,5)
        self.bn1 = BatchNorm(16)
        self.bn_img = BatchNorm(16)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,16,5)
        self.bn2 = BatchNorm(16)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(16,1,3)


    def forward(self,x):
        d = x['d']
        img = x['rgb']
        ip0 = img.shape[-2]
        ip1 = img.shape[-1]
        x = F.relu(self.bn1(self.conv1(d)))
        img = F.relu(self.bn_img(self.conv_img(img)))
        concat_feat = torch.cat((x,img),dim=1)
        x = self.pool2(F.relu(self.bn2(self.conv2(concat_feat))))
        x = F.relu(self.conv3(x))
        x = nn.Upsample(size=(ip0//2,ip1//2),mode='bilinear',align_corners=True)(x)
        x = nn.Upsample(size=(ip0,ip1),mode='bilinear',align_corners=True)(x)
        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                 m.weight.data.normal_(0, math.sqrt(2. / n))
                #nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class UpProj(Decoder):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    class UpProjModule(nn.Module):
        # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
        #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
        #   bottom branch: 5*5 conv -> batchnorm

        def __init__(self, in_channels):
            super(UpProj.UpProjModule, self).__init__()
            out_channels = in_channels//2
            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(collections.OrderedDict([
              ('conv1',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
              ('batchnorm1', nn.BatchNorm2d(out_channels)),
              ('relu',      nn.ReLU()),
              ('conv2',      nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)),
              ('batchnorm2', nn.BatchNorm2d(out_channels)),
            ]))
            self.bottom_branch = nn.Sequential(collections.OrderedDict([
              ('conv',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
              ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.unpool(x)
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, in_channels):
        super(UpProj, self).__init__()
        self.layer1 = self.UpProjModule(in_channels)
        self.layer2 = self.UpProjModule(in_channels//2)
        self.layer3 = self.UpProjModule(in_channels//4)
        self.layer4 = self.UpProjModule(in_channels//8)

def choose_decoder(decoder, in_channels):
    # iheight, iwidth = 10, 8
    if decoder[:6] == 'deconv':
        assert len(decoder)==7
        kernel_size = int(decoder[6])
        return DeConv(in_channels, kernel_size)
    elif decoder == "upproj":
        return UpProj(in_channels)
    elif decoder == "upconv":
        return UpConv(in_channels)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)


class ResNet(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(ResNet, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.output_size = output_size

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels//2)
        self.decoder = choose_decoder(decoder, num_channels//2)

        # setting bias=true doesn't improve accuracy
        self.conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # decoder
        x = self.decoder(x)
        x = self.conv3(x)
        x = self.bilinear(x)

        return x

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def conv_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
        padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

def convt_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
        stride, padding, output_padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

class DepthCompletionNet(nn.Module):
    def __init__(self, args):
        assert (args.layers in [18, 34, 50, 101, 152]), 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(layers)
        super(DepthCompletionNet, self).__init__()
        self.modality = 'rgbd'

        if 'd' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_d = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)
        if 'rgb' in self.modality:
            channels = 64 * 2 // len(self.modality)
            self.conv1_img = conv_bn_relu(3, channels, kernel_size=3, stride=1, padding=1)
        elif 'g' in self.modality:
            channels = 64 // len(self.modality)
            self.conv1_img = conv_bn_relu(1, channels, kernel_size=3, stride=1, padding=1)
            

        self.conv1_normal = conv_bn_relu(3, 64//len(self.modality), kernel_size=3, stride=1, padding=1)

        pretrained_model = resnet.__dict__['resnet{}'.format(args.layers)](pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_model.apply(init_weights)
        if args.pretrained:
            print("using pretrained weights")
        #self.maxpool = pretrained_model._modules['maxpool']
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model # clear memory

        # define number of intermediate channels
        if args.layers <= 34:
            num_channels = 512
        elif args.layers >= 50:
            num_channels = 2048
        self.conv6 = conv_bn_relu(num_channels, 512, kernel_size=3, stride=2, padding=1)

        # decoding layers
        kernel_size = 3
        stride = 2
        self.convt5 = convt_bn_relu(in_channels=512, out_channels=256,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=768, out_channels=128,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=(256+128), out_channels=64,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=(128+64), out_channels=64,
            kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=128, out_channels=64,
            kernel_size=kernel_size, stride=1, padding=1)
        self.convtf = conv_bn_relu(in_channels=128, out_channels=1, kernel_size=1, stride=1, bn=False, relu=False)

    def forward(self, x, N):
        # first layer
        if 'd' in self.modality:
            conv1_d = self.conv1_d(x['d'])
        if 'rgb' in self.modality:
            conv1_img = self.conv1_img(x['rgb'])
        elif 'g' in self.modality:
            conv1_img = self.conv1_img(x['g'])

        conv1_normal = self.conv1_normal(N)
        if self.modality=='rgbd' or self.modality=='gd':
            conv1 = torch.cat((conv1_d, conv1_img, conv1_normal),1)
        else:
            conv1 = conv1_d if (self.modality=='d') else conv1_img

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2) # batchsize * ? * 176 * 608
        conv4 = self.conv4(conv3) # batchsize * ? * 88 * 304
        conv5 = self.conv5(conv4) # batchsize * ? * 44 * 152
        conv6 = self.conv6(conv5) # batchsize * ? * 22 * 76

        # decoder
        convt5 = self.convt5(conv6)
        convt5 = torch.nn.functional.upsample(convt5,size=(conv5.shape[-2],conv5.shape[-1]))
        y = torch.cat((convt5, conv5), 1)

        convt4 = self.convt4(y)
        convt4 = torch.nn.functional.upsample(convt4,size=(conv4.shape[-2],conv4.shape[-1]))
        y = torch.cat((convt4, conv4), 1)

        convt3 = self.convt3(y)
        convt3 = torch.nn.functional.upsample(convt3,size=(conv3.shape[-2],conv3.shape[-1]))
        y = torch.cat((convt3, conv3), 1)

        convt2 = self.convt2(y)
        convt2 = torch.nn.functional.upsample(convt2,size=(conv2.shape[-2],conv2.shape[-1]))
        y = torch.cat((convt2, conv2), 1)

        convt1 = self.convt1(y)
        convt1 = torch.nn.functional.upsample(convt1,size=(conv1.shape[-2],conv1.shape[-1]))
        y = torch.cat((convt1,conv1), 1)

        y = self.convtf(y)

        return y
        #if self.training:
        #    return 100 * y
        #else:
        #    min_distance = 0.9
        #    return F.relu(100 * y - min_distance) + min_distance # the minimum range of Velodyne is around 3 feet ~= 0.9m


class ERF(nn.Module):
    def __init__(self, encoder=None):  # use encoder to pass pretrained encoder
        super().__init__()

        self.config = None
 
        self.num_classes = 1
        self.input_channels = 4
            

        self.conv1_d = conv_bn_relu(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_img = conv_bn_relu(3, 128, kernel_size=3, stride=1, padding=1)
    
        if encoder == None:
            self.encoder_flag = True
            self.encoder_layers = nn.ModuleList()

            # layer 1, downsampling
            self.initial_block = DownsamplerBlock(self.input_channels, 16)

            # layer 2, downsampling
            self.encoder_layers.append(DownsamplerBlock(in_channel=16, out_channel=64))

            # non-bottleneck 1d - layers 3 to 7
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))

            # layer 8, downsampling
            self.encoder_layers.append(DownsamplerBlock(in_channel=64, out_channel=128))

            # non-bottleneck 1d - layers 9 to 16
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=2))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=4))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=8))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=16))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=2))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=4))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=8))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=16))

        else:
            self.encoder_flag = False
            self.encoder = encoder

        self.decoder_layers = nn.ModuleList()

        self.decoder_layers.append(UpsamplerBlock(in_channel=128, out_channel=64,output_padding=(0,1)))
        self.decoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0, dilated=1))
        self.decoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0, dilated=1))

        self.decoder_layers.append(UpsamplerBlock(in_channel=64, out_channel=16))
        self.decoder_layers.append(non_bottleneck_1d(n_channel=16, drop_rate=0, dilated=1))
        self.decoder_layers.append(non_bottleneck_1d(n_channel=16, drop_rate=0, dilated=1))

        self.output_conv = nn.ConvTranspose2d(in_channels=16, out_channels=self.num_classes,kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

        # self.apply(weights_init_normal)

    def forward(self, x):

        #conv1_d = self.conv1_d(x['d'])
        #conv1_img = self.conv1_img(x['rgb'])
        #conv1 = torch.cat((conv1_d, conv1_img),1)
        conv1 = torch.cat((x['d'], x['rgb']),1)
        if self.encoder_flag:
            output = self.initial_block(conv1)
            for layer in self.encoder_layers:
                output = layer(output)
        else:
            output = self.encoder(conv1)

        for layer in self.decoder_layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

class ERF_N(nn.Module):
    def __init__(self, encoder=None):  # use encoder to pass pretrained encoder
        super().__init__()

        self.config = None
 
        self.num_classes = 3
        self.input_channels = 3
            

        self.conv1_d = conv_bn_relu(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_img = conv_bn_relu(3, 128, kernel_size=3, stride=1, padding=1)
    
        if encoder == None:
            self.encoder_flag = True
            self.encoder_layers = nn.ModuleList()

            # layer 1, downsampling
            self.initial_block = DownsamplerBlock(self.input_channels, 16)

            # layer 2, downsampling
            self.encoder_layers.append(DownsamplerBlock(in_channel=16, out_channel=64))

            # non-bottleneck 1d - layers 3 to 7
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0.03, dilated=1))

            # layer 8, downsampling
            self.encoder_layers.append(DownsamplerBlock(in_channel=64, out_channel=128))

            # non-bottleneck 1d - layers 9 to 16
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=2))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=4))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=8))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=16))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=2))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=4))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=8))
            self.encoder_layers.append(non_bottleneck_1d(n_channel=128, drop_rate=0.3, dilated=16))

        else:
            self.encoder_flag = False
            self.encoder = encoder

        self.decoder_layers = nn.ModuleList()

        self.decoder_layers.append(UpsamplerBlock(in_channel=128, out_channel=64,output_padding=(0,1)))
        self.decoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0, dilated=1))
        self.decoder_layers.append(non_bottleneck_1d(n_channel=64, drop_rate=0, dilated=1))

        self.decoder_layers.append(UpsamplerBlock(in_channel=64, out_channel=16))
        self.decoder_layers.append(non_bottleneck_1d(n_channel=16, drop_rate=0, dilated=1))
        self.decoder_layers.append(non_bottleneck_1d(n_channel=16, drop_rate=0, dilated=1))

        self.output_conv = nn.ConvTranspose2d(in_channels=16, out_channels=self.num_classes,kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

        # self.apply(weights_init_normal)

    def forward(self, x):

        #conv1_d = self.conv1_d(x['d'])
        #conv1_img = self.conv1_img(x['rgb'])
        #conv1 = torch.cat((conv1_d, conv1_img),1)
        #conv1 = torch.cat((x['d'], x['rgb']),1)
        if self.encoder_flag:
            output = self.initial_block(x['rgb'])
            for layer in self.encoder_layers:
                output = layer(output)
        else:
            output = self.encoder(conv1)

        for layer in self.decoder_layers:
            output = layer(output)

        output = nn.Tanh()(self.output_conv(output))

        return output

