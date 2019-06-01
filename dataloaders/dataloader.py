import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py
import cv2
import dataloaders.transforms as transforms
from estimate_pose import get_pose

IMG_EXTENSIONS = ['.h5',]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth

def near_rgb(path,loader):

    filename = os.path.basename(path)
    frame_id = filename.split(".")[0]
    file_ext = filename.split(".")[1]
    filepath = os.path.split(path)[0]
    str_len = len(frame_id)
    window = 6
    found = False
    paths_tested = []
    for k in range(-window,window):
        if (k==0):
            continue
        fid = int(frame_id)+k
        fid_str = str(fid).zfill(str_len)
        new_file_path = os.path.join(filepath,fid_str+"."+file_ext)
        paths_tested.append(new_file_path)
        if (os.path.exists(new_file_path)):
            found = True
            break
        
    if (found):
       rgb,depth = h5_loader(new_file_path)
       return rgb
    else:
       print("Near not found for ", path, " new = ", paths_tested)
       return None 

# def rgb2grayscale(rgb):
#     return rgb[:,:,0] * 0.2989 + rgb[:,:,1] * 0.587 + rgb[:,:,2] * 0.114

to_tensor = transforms.ToTensor()

to_float_tensor = lambda x: to_tensor(x).float()

class MyDataloader(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd'] # , 'g', 'gd'
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, type, sparsifier=None, modality='rgb', loader=h5_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        assert len(imgs)>0, "Found 0 images in subfolders of: " + root + "\n"
        print("Found {} images in {} folder.".format(len(imgs), type))
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.mode = type
        if type == 'train':
            self.transform = self.train_transform
        elif type == 'val':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))
        self.loader = loader
        self.sparsifier = sparsifier

        self.K = None
        self.output_size = None

        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality

    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    def create_sparse_depth(self, rgb, depth):
        if self.sparsifier is None:
            return depth
        else:
            mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
            sparse_depth = np.zeros(depth.shape)
            sparse_depth[mask_keep] = depth[mask_keep]
            return sparse_depth

    def create_rgbd(self, rgb, depth):
        sparse_depth = self.create_sparse_depth(rgb, depth)
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        return rgbd

    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path, target = self.imgs[index]
        rgb, depth = self.loader(path)
        rgb_near = near_rgb(path,self.loader)
        return rgb, depth, rgb_near

    def __getitem__(self, index):
        rgb, depth, rgb_near = self.__getraw__(index)
        if self.transform is not None:
            rgb_np, depth_np, rgb_near_np = self.transform(rgb, depth,rgb_near)
        else:
            raise(RuntimeError("transform not defined"))

        # If in train mode, compute pose for near image
        #print("K = ", self.K)
        rot_mat, t_vec = None, None
        if self.mode == "train":
            rgb_cv = (rgb_np*255).astype(np.uint8)
            rgb_near_cv = (rgb_near_np*255).astype(np.uint8)
            succ, rot_vec, t_vec = get_pose(rgb_cv, depth_np, rgb_near_cv, self.K)
            if succ:
                rot_mat, _ = cv2.Rodrigues(rot_vec)
            else:
                rgb_near_np = rgb_np
                t_vec = np.zeros((3,1))
                rot_mat = np.eye(3)
            
        # color normalization
        # rgb_tensor = normalize_rgb(rgb_tensor)
        # rgb_np = normalize_np(rgb_np)

        if self.modality == 'rgb':
            input_np = rgb_np
        elif self.modality == 'rgbd':
            input_np = self.create_rgbd(rgb_np, depth_np)
        elif self.modality == 'd':
            input_np = self.create_sparse_depth(rgb_np, depth_np)


        #input_tensor = to_tensor(input_np)
        #while input_tensor.dim() < 3:
        #    input_tensor = input_tensor.unsqueeze(0)
        #depth_tensor = to_tensor(depth_np)
        #depth_tensor = depth_tensor.unsqueeze(0)
        #rgb_near_tensor = to_tensor(rgb_near)
        #print(input_np.shape) 
        candidates = {"rgb":rgb_np, "gt":np.expand_dims(depth_np,-1), "d":input_np[:,:,3:], \
                      "r_mat":rot_mat, "t_vec":t_vec, "rgb_near":rgb_near_np}


        #print(self.K)
        intrinsics = {"fx":self.K[0,0],
                      "fy":self.K[1,1],
                      "cx":self.K[0,2],
                      "cy":self.K[1,2],
                      "output_size" : self.output_size}
        items = {key:to_float_tensor(val) for key, val in candidates.items() if val is not None}

        
        return items, intrinsics
        
        #return input_tensor, depth_tensor

    def __len__(self):
        return len(self.imgs)

    # def __get_all_item__(self, index):
    #     """
    #     Args:
    #         index (int): Index

    #     Returns:
    #         tuple: (input_tensor, depth_tensor, input_np, depth_np)
    #     """
    #     rgb, depth = self.__getraw__(index)
    #     if self.transform is not None:
    #         rgb_np, depth_np = self.transform(rgb, depth)
    #     else:
    #         raise(RuntimeError("transform not defined"))

    #     # color normalization
    #     # rgb_tensor = normalize_rgb(rgb_tensor)
    #     # rgb_np = normalize_np(rgb_np)

    #     if self.modality == 'rgb':
    #         input_np = rgb_np
    #     elif self.modality == 'rgbd':
    #         input_np = self.create_rgbd(rgb_np, depth_np)
    #     elif self.modality == 'd':
    #         input_np = self.create_sparse_depth(rgb_np, depth_np)

    #     input_tensor = to_tensor(input_np)
    #     while input_tensor.dim() < 3:
    #         input_tensor = input_tensor.unsqueeze(0)
    #     depth_tensor = to_tensor(depth_np)
    #     depth_tensor = depth_tensor.unsqueeze(0)

    #     return input_tensor, depth_tensor, input_np, depth_np
