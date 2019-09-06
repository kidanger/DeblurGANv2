import random
import torch
import torch.utils.data as data

import utils as U

from generate_kernel import generate_kernel

import iio
from skimage.transform import resize
import functools
from skimage.io import imread

def get_image(file):
    img = imread(file) / 255
    # img = resize(img, (img.shape[0]//1.5,img.shape[1]//1.5), order=0, mode='reflect',
                 # anti_aliasing=True, anti_aliasing_sigma=(0.6,0.6,0), clip=False)
    # img = img.clip(0, 1)
    img = torch.FloatTensor(img)
    img = img.permute((2,0,1))
    return img

class SyntheticDatasetFromFiles(data.Dataset):

    def __init__(self, images, transform, val=False):
        super(SyntheticDatasetFromFiles, self).__init__()
        from os import listdir
        from os.path import join
        self.image_filenames = images
        self.transform = transform or (lambda x, k: x, x)
        self.val = val
        self.kernels = []
        self.random = random.Random(0)
        if val:
            for i in range(100):
                ks = self.random.randint(9, 51)
                ks += (ks+1) % 2
                k = generate_kernel(ks, rand=self.random)
                k = torch.FloatTensor(k, device='cpu')
                k = k.unsqueeze(0)
                self.kernels.append(k)

    def __getitem__(self, index):
        with torch.no_grad():
            if self.kernels:
                k = self.kernels[index % len(self.kernels)]
            else:
                ks = self.random.randint(9, 51)
                ks += (ks+1) % 2
                k = generate_kernel(ks)
                k = torch.FloatTensor(k, device='cpu')
                k = k.unsqueeze(0)
            input = get_image(self.image_filenames[index % len(self.image_filenames)])
            crop = center_crop if self.val else random_crop
            input = crop(input, 256 + max(k.size()))
            input, target = self.transform(input, k, train=not self.val)
            # if random.random()<0.01:
                # U.write_tensor('input.tif', input.unsqueeze(0))
                # U.write_tensor('target.tif', target.unsqueeze(0))
        return {'a': input, 'b': target}

    def __len__(self):
        if self.val:
            return len(self.image_filenames)
        return len(self.image_filenames) * 100

def blur(img, k):
    out = torch.nn.functional.conv2d(img.unsqueeze(0), k.expand((3,1,-1,-1)), padding=max(k.size())//2, bias=None, groups=3)
    return out.squeeze(0)

def add_noise(img, sigma, concat_noise=False):
    if sigma < 0:
        sigma = random.random() * -sigma
    v = img + torch.randn_like(img) * sigma
    if concat_noise:
        noise_map = torch.FloatTensor([sigma], device='cpu')
        noise_map = noise_map.repeat(1, *v.shape[1:])
        v = torch.cat((v, noise_map), 0)
    return v

def split_noise(v):
    v, sigma = v[..., :-1, :, :], v[..., -1:, :, :]
    sigma = sigma[..., 0, 0, 0]
    return v, sigma

def center_crop(img, crop_size):
    size = img.size()
    w, h = size[-2], size[-1]
    return img[...,(w-crop_size)//2:(w+crop_size)//2,(h-crop_size)//2:(h+crop_size)//2]

def random_crop(img, crop_size):
    size = img.size()
    w, h = size[-2], size[-1]
    x = random.randint(0, w - crop_size)
    y = random.randint(0, h - crop_size)
    return img[...,x:x+crop_size,y:y+crop_size]

def get_transform(crop_size, sigma, circular=True, concat_noise=False):
    def f(img, k, train):
        # crop = random_crop if train else center_crop
        crop = center_crop
        if circular:
            img = crop(img, crop_size)
            b = blur(img, k)
        else:
            b = blur(img, k)
            img = crop(img, crop_size)
            b = crop(b, crop_size)
        b = add_noise(b, sigma, concat_noise=concat_noise)
        return b, img
    return f

