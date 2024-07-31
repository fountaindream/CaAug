import torch
import functools
import os.path as osp
import torchvision.transforms as T
import numpy as np
import random
import collections
from PIL import Image
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.beta = 1.0
        self.transform1 = T.Compose([T.Resize((384, 128), interpolation=3),T.RandomApply([T.ColorJitter(0, 0, 0, 0.3)], p=0.5),T.ToTensor()])

    def __len__(self):
        return len(self.dataset)


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.T = T.Resize((384,192))
        self.color = T.RandomApply([T.ColorJitter(0, 0, 0, 0.3)], p=0.5)
        self.data_dict = Data_dict(dataset)
        self.beta = 1.0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]       
        img = read_image(img_path)
        img_sampled = self.sample_image(pid)
        img, img_sampled = self.T(img),self.T(img_sampled)
        img1 = self.cutmix(img, img_sampled)
        img2 = self.mixup(img, img_sampled)
        img3 = self.cutout(img)      
        img4 = self.color(img)
        if self.transform is not None:
            img = self.transform(img)
            img1 = self.transform(img1)             
            img2 = self.transform(img2) 
            img3 = self.transform(img3)
            img4 = self.transform(img4)                                 
        return img, [img1,img2,img3,img4], pid, camid, clothes_id  
    
    def sample_image(self,key):
        img_idx = random.randint(0, len( self.data_dict[key])-1)
        img_sampled = read_image(self.data_dict[key][img_idx])

        return img_sampled

    def rand_bbox(self, size, lam):
        if len(size) == 4:
            W = size[2]
            H = size[3]
        elif len(size) == 3:
            W = size[1]
            H = size[2]
        else:
            raise Exception

        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def mixup(self, img, img2):
        img = np.array(img)
        img2 = np.array(img2)
        lam = np.random.beta(self.beta, self.beta)
        img = lam * img + (1 - lam) * img2
        img = Image.fromarray(np.uint8(np.clip(img, 0, 255)))
        return img

    def cutout(self, img):
        img = np.array(img)
        lam = np.random.beta(self.beta, self.beta)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(img.shape, lam)
        img[:, bbx1:bbx2, bby1:bby2] = 0
        img = Image.fromarray(np.uint8(np.clip(img, 0, 255)))
        return img

    def cutmix(self, img, img2):
        img = np.array(img)
        img2 = np.array(img2)
        lam = np.random.beta(self.beta, self.beta)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(img.shape, lam)
        img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
        img = Image.fromarray(np.uint8(np.clip(img, 0, 255)))
        return img

def Data_dict(dataset):
    dataset_dict = collections.defaultdict(list)
    for img_path, pid, _, _ in dataset:
        dataset_dict[pid].append(img_path)
    return dataset_dict 

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def video_loader(img_paths, image_loader):
    video = []
    for image_path in img_paths:
        if osp.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self, 
                 dataset, 
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader,
                 cloth_changing=True):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.cloth_changing = cloth_changing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        if self.cloth_changing:
            img_paths, pid, camid, clothes_id = self.dataset[index]
        else:
            img_paths, pid, camid = self.dataset[index]

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)

        clip = self.loader(img_paths)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        # trans T x C x H x W to C x T x H x W
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        if self.cloth_changing:
            return clip, pid, camid, clothes_id
        else:
            return clip, pid, camid