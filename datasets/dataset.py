import os
import random
from PIL import Image
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        # print("IMAGE SHAPE: ", image.shape)
        x, y, _ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        image = image.permute(2, 0, 1)

        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class Building_dataset(Dataset):
    def __init__(self, args, split, transform=None):
        self.base_dir = args.root_path
        self.img_size = args.img_size
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(self.base_dir, self.split+'.txt')).readlines()

    def __len__(self):
        return len(self.sample_list)

    def convert_to_binary(self, label):
        label[label != 29] = 0 # 29 is the building class pixel value
        label[label == 29] = 255
        return label

    def __getitem__(self, idx):
        if self.split == 'train':
            file_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.base_dir, 'images', file_name)
            img = np.array(Image.open(data_path).convert('RGB')).astype(np.float32) / 255.0
            label_path = os.path.join(self.base_dir, 'labels', file_name)
            gt = self.convert_to_binary(np.array(Image.open(label_path).convert('L'))).astype(np.float32) / 255.0

            sample = {'image': img, 'label': gt}
            if self.transform:
                sample = self.transform(sample)
        elif self.split == 'val':
            file_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.base_dir, 'images', file_name)
            img = np.array(Image.open(data_path).convert('RGB')).astype(np.float32) / 255.0
            label_path = os.path.join(self.base_dir, 'labels', file_name)
            gt = self.convert_to_binary(np.array(Image.open(label_path).convert('L'))).astype(np.float32) / 255.0
            x, y, _ = img.shape
            if x != self.img_size or y != self.img_size:
                img = zoom(img, (self.img_size / x, self.img_size / y, 1), order=3)
                gt = zoom(gt, (self.img_size / x, self.img_size / y), order=0)
            sample = {'image': torch.from_numpy(img).permute(2, 0, 1), 'label': torch.from_numpy(gt)}
        return sample
