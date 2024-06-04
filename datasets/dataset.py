import os
import random
from PIL import Image
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset
from utils import convert_to_binary


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
        assert x == self.output_size and y == self.output_size, f'Image size {x}*{y} not equal to output size {self.output_size}, please check!'
        image = torch.from_numpy(image.astype(np.float32))
        image = image.permute(2, 0, 1)

        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class Building_dataset(Dataset):
    """
    Args:
        args: configuration parameters
        split: 'train', 'val' or "test"
        transform: data augmentation
    """
    def __init__(self, args, split, transform=None):
        self.base_dir = args.root_path
        self.img_size = args.img_size
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(self.base_dir, self.split+'.txt')).readlines()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == 'train':
            file_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.base_dir, 'images', file_name)
            img = np.array(Image.open(data_path).convert('RGB')).astype(np.float32) / 255.0
            label_path = os.path.join(self.base_dir, 'labels', file_name)
            gt = convert_to_binary(np.array(Image.open(label_path).convert('L'))).astype(np.float32) / 255.0

            sample = {'image': img, 'label': gt}
            if self.transform:
                sample = self.transform(sample)
        elif self.split == 'val':
            file_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.base_dir, 'images', file_name)
            img = np.array(Image.open(data_path).convert('RGB')).astype(np.float32) / 255.0
            label_path = os.path.join(self.base_dir, 'labels', file_name)
            gt = convert_to_binary(np.array(Image.open(label_path).convert('L'))).astype(np.float32) / 255.0
            x, y, _ = img.shape
            assert x == self.img_size and y == self.img_size, f'Image size {x}*{y} not equal to output size {self.img_size}, please check!'
            sample = {'image': torch.from_numpy(img).permute(2, 0, 1), 'label': torch.from_numpy(gt)}
        elif self.split == 'test':
            file_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.base_dir, 'images', file_name)
            img = np.array(Image.open(data_path).convert('RGB')).astype(np.float32) / 255.0
            label_path = os.path.join(self.base_dir, 'labels', file_name)
            gt = convert_to_binary(np.array(Image.open(label_path).convert('L'))).astype(np.float32) / 255.0
            x, y, _ = img.shape
            assert x == self.img_size and y == self.img_size, f'Image size {x}*{y} not equal to output size {self.img_size}, please check!'
            sample = {'image': torch.from_numpy(img).permute(2, 0, 1), 'label': torch.from_numpy(gt), 'case_name': file_name}
        else:
            raise NotImplementedError
        return sample
