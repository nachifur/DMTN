import glob
import os
import random

import Augmentor
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from skimage.color import rgb2gray

from .utils import imshow, create_mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, data_flist, mask_flist, GT_flist, additional_mask=[], augment=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.data = self.load_flist(data_flist)
        self.mask = self.load_flist(mask_flist)
        self.GT = self.load_flist(GT_flist)
        if len(additional_mask) != 0:
            self.additional_mask = self.load_flist(additional_mask)
        else:
            self.additional_mask = []

        self.input_size_h = config.INPUT_SIZE_H
        self.input_size_w = config.INPUT_SIZE_W
        self.mask_type = config.MASK
        self.dataset_name = config.SUBJECT_WORD

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size_h = self.input_size_h
        size_w = self.input_size_w

        # load image
        img = cv2.imread(self.data[index], -1)
        mask_GT = cv2.imread(self.mask[index], -1)
        GT = cv2.imread(self.GT[index], -1)

        # augment data
        if self.augment:
            img, mask_GT, GT = self.data_augment(
                img, mask_GT, GT)
        else:
            imgh, imgw = img.shape[0:2]
            if not (size_h == imgh and size_w == imgw):
                img = self.resize(img, size_h, size_w)
                mask_GT = self.resize(mask_GT, size_h, size_w)
                GT = self.resize(GT, size_h, size_w)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        GT = cv2.cvtColor(GT, cv2.COLOR_BGR2RGB)

        # imshow(Image.fromarray(edge_truth))
        return self.to_tensor(img), self.to_tensor(mask_GT), self.to_tensor(GT)

    def data_augment(self, img, mask_GT, GT):

        # https://github.com/mdbloice/Augmentor
        images = [[img, GT, mask_GT]]
        p = Augmentor.DataPipeline(images)
        p.flip_random(1)
        width=random.randint(256,640)
        height = round((width/self.input_size_w)*self.input_size_h)
        p.resize(1, width, height)
        g = p.generator(batch_size=1)
        augmented_images = next(g)
        images = augmented_images[0][0]
        GT = augmented_images[0][1]
        mask_GT = augmented_images[0][2]

        if len(self.additional_mask) != 0:
            additional_mask = self.load_mask(
                self.input_size_h, self.input_size_w, random.randint(0, len(self.additional_mask)))
            additional_mask = additional_mask.astype(
                np.single)/np.max(additional_mask)*np.random.uniform(0, 0.8)
            additional_mask = additional_mask[:, :, np.newaxis]
            images = images*(1-additional_mask)
            images = images.astype(np.uint8)
        return images, mask_GT, GT

    def load_mask(self, imgh, imgw, index):
        mask_type = self.mask_type

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # half
        if mask_type == 2:
            # randomly choose right or left
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.additional_mask) - 1)
            mask = cv2.imread(self.additional_mask[mask_index])
            mask = 255-mask
            mask = self.resize(mask, imgh, imgw)
            # threshold due to interpolation
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = cv2.imread(self.additional_mask[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)  # returns an image object.
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + \
                    list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
