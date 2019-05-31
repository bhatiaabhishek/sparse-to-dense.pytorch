import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader

iheight, iwidth = 480, 640 # raw image size
# RGB Intrinsic Parameters
fx_rgb = 5.1885790117450188e+02;
fy_rgb = 5.1946961112127485e+02;
cx_rgb = 3.2558244941119034e+02;
cy_rgb = 2.5373616633400465e+02;

#After 0.5*random sclaing, there is a center crop which of 228,304 which will change cx and cy
def TransfromIntrinsics(K,scale,output_size): 


    crop_h,crop_w = output_size
    fx_s = fx_rgb*scale
    fy_s = fy_rgb*scale
    cx_s = cx_rgb*scale
    cy_s = cy_rgb*scale

    cut_x = ((iwidth*scale)-crop_w)//2 # Cut on one side
    cut_y = ((iheight*scale)-crop_h)//2

    cx_s = cx_s - cut_x
    cy_s = cy_s - cut_y
    #print("scale = ", scale, ", cutx = ", cut_x, ", cuty = ", cut_y)
    return np.array([[fx_s, 0, cx_s],
                     [0, fy_s, cy_s],
                     [0,0,1]])

class NYUDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb'):
        super(NYUDataset, self).__init__(root, type, sparsifier, modality)
        self.output_size = (228, 304)
        self.K = np.array([[fx_rgb, 0, cx_rgb],
                           [0, fy_rgb, cy_rgb],
                           [0, 0, 1]])

    def train_transform(self, rgb, depth,rgb_near):
        s = np.random.uniform(1.0, 1.5) # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip
        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(250.0 / iheight), # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        rgb_near_np = None
        if rgb_near is not None:
            rgb_near_np = transform(rgb_near)
            rgb_near_np = np.asfarray(rgb_near_np,dtype='float') / 255
        depth_np = transform(depth_np)

        self.K = TransfromIntrinsics(self.K,(250.0/iheight)*s,self.output_size)
        return rgb_np, depth_np, rgb_near_np

    def val_transform(self, rgb, depth,rgb_near):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize(240.0 / iheight),
            transforms.CenterCrop(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        rgb_near_np = None
        if rgb_near is not None:
            rgb_near_np = transform(rgb_near)
            rgb_near_np = np.asfarray(rgb_near_np, dtype='float') / 255
        depth_np = transform(depth_np)
        self.K = TransfromIntrinsics(self.K,(240.0/iheight),self.output_size)

        return rgb_np, depth_np, rgb_near_np
