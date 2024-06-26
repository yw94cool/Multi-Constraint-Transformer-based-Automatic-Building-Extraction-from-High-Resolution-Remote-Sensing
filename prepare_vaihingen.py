import argparse
import numpy as np
import os
from PIL import Image
from random import sample

def generate_list(in_dir: str, mode='fix'):
    """
    Randomly split the dataset into training and validation set.
    Args:
        in_dir: str, the directory of the dataset after being prepared.
        mode: str, 'fix' or 'random', if 'fix', the split will be fixed, otherwise, it will be random.
        (To reprodce the results in our paper, please use 'fix' mode.)
    """
    all_files = os.listdir(os.path.join(in_dir, 'images'))
    if mode == 'fix':
        test_files = [file for file in all_files if 'area37' in file]
        val_files = [file for file in all_files if 'area32' in file or 'area34' in file]
        train_files = [file for file in all_files if file not in test_files and file not in val_files]
    elif mode == 'random':
        train_files = sample(all_files, int(0.7 * len(all_files)))
        val_files = sample([file for file in all_files if file not in train_files], int(0.2 * len(all_files)))
        test_files = [file for file in all_files if file not in train_files and file not in val_files]
    else:
        raise ValueError("Invalid mode, please input 'fix' or 'random'!")
    with open(os.path.join(in_dir, 'train.txt'), 'w') as f:
        for file in train_files:
            f.write('{}\n'.format(file))
    with open(os.path.join(in_dir, 'val.txt'), 'w') as f:
        for file in val_files:
            f.write('{}\n'.format(file))
    with open(os.path.join(in_dir, 'test.txt'), 'w') as f:
        for file in test_files:
            f.write('{}\n'.format(file))

def main(args):
    tar_path = args.data_dir
    new_dir = os.path.join(os.path.dirname(tar_path), 'Prepared_Vaihingen')
    for folder in ['images', 'labels']:
        os.makedirs(os.path.join(new_dir, folder), exist_ok=True)

    # Crop images and labels
    print('Cropping images and labels...')
    for file in os.listdir(os.path.join(tar_path, 'top')):
        if file.endswith('.tif'):
            img = np.array(Image.open(os.path.join(tar_path, 'top', file)))
            label = np.array(Image.open(os.path.join(tar_path, 'groundtruth', file)))
            assert img.shape == label.shape, f'Image and Gt of {file}  have different shape, please check your dataset!'

            patch_size = args.patch_size
            height, width = img.shape[:2]
            for i in range(0, height, patch_size):
                for j in range(0, width, patch_size):
                    if i + patch_size > height or j + patch_size > width:
                        continue
                    patch_img = img[i:i+patch_size, j:j+patch_size]
                    patch_label = label[i:i+patch_size, j:j+patch_size]
                    patch_img = Image.fromarray(patch_img)
                    patch_label = Image.fromarray(patch_label)
                    patch_img.save(os.path.join(new_dir, 'images', '{}_{}_{}.tif'.format(file.split('.')[0], i // patch_size, j // patch_size)))
                    patch_label.save(os.path.join(new_dir, 'labels', '{}_{}_{}.tif'.format(file.split('.')[0], i // patch_size, j // patch_size)))
    
    generate_list(new_dir)
    print('Dataset prepared successfully!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/Vaihingen')
    parser.add_argument('--patch_size', type=int, default=256)
    args = parser.parse_args()
    main(args)