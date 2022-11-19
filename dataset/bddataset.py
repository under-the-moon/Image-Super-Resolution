import os
import numpy as np
import cv2
import torch
import glob
from torch.utils.data import Dataset


class ISRDataset(Dataset):

    def __init__(self, data_path, scale=2, img_size=32, num_patchs=64, shuffle=True):
        self.img_size = img_size
        self.scale = scale
        self.num_patchs = num_patchs
        image_paths = self._load_data(data_path)
        if shuffle:
            np.random.shuffle(image_paths)
        self.image_paths = image_paths

    def _load_data(self, data_path):
        img_paths = glob.glob(os.path.join(data_path, 'train', '*', 'x', '*'))
        if self.scale == 2:
            img_paths2 = glob.glob(os.path.join(data_path, 'train', '*', 'x2', '*'))
            img_paths.extend(img_paths2)
        return img_paths

    def __len__(self):
        return len(self.image_paths)

    def pad_img(self, img, scale=1.):
        H, W, C = img.shape
        img_size = self.img_size * scale
        pad_h = img_size - (H % img_size)
        pad_w = img_size - (W % img_size)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)))
        return img

    def patch_embed(self, img, scale=1):
        h, w, c = img.shape
        img_size = self.img_size * scale
        img = img.reshape(h // img_size, img_size, w // img_size, img_size, c)
        img = img.transpose((0, 2, 1, 3, 4))
        img = img.reshape((-1, img_size, img_size, c))
        return img

    def __getitem__(self, item):
        img_path = self.image_paths[item]

        high_img_path = img_path.replace('x', 'x{}'.format(self.scale))
        if self.scale == 2 and 'x2' in img_path:
            high_img_path = img_path.replace('x2', 'x4')

        low_img = cv2.imread(img_path)
        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        h_img = cv2.imread(high_img_path)
        h_img = cv2.cvtColor(h_img, cv2.COLOR_BGR2RGB)

        low_img = self.pad_img(low_img, scale=1)
        h_img = self.pad_img(h_img, scale=2)

        low_img = self.patch_embed(low_img, scale=1)
        h_img = self.patch_embed(h_img, scale=2)

        indics = np.random.choice([i for i in range(len(low_img))], size=self.num_patchs)

        low_img = low_img[indics, ...]
        h_img = h_img[indics, ...]

        img = low_img.copy()
        img = img.transpose(0, 3, 1, 2)

        target = h_img.copy()
        target = target.transpose(0, 3, 1, 2)

        img = img / 255
        target = target / 255
        return img, target


def collate_fn(batch):
    batch_img = [data[0] for data in batch]
    batch_target = [data[1] for data in batch]

    batch_img = np.array(batch_img)
    batch_target = np.array(batch_target)

    batch_img = torch.from_numpy(batch_img)
    batch_target = torch.from_numpy(batch_target)

    batch_img = batch_img.reshape(-1, batch_img.shape[2], batch_img.shape[3], batch_img.shape[4])
    batch_target = batch_target.reshape(-1, batch_target.shape[2], batch_target.shape[3], batch_target.shape[4])
    return batch_img, batch_target