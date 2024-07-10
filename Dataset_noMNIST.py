from datetime import datetime
from functools import partial
import cv2
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import DataLoader,TensorDataset, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, FashionMNIST,MNIST
from torchvision.models import resnet
from tqdm import tqdm
import argparse
import json
import math
import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomGaussianBlur(object):
    def __init__(self, p=0.5, min_kernel_size=3, max_kernel_size=15, min_sigma=0.1, max_sigma=1.0):
        self.p = p
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img):
        if random.random() < self.p and self.min_kernel_size<self.max_kernel_size:
            kernel_size = random.randrange(self.min_kernel_size, self.max_kernel_size+1, 2)
            sigma = random.uniform(self.min_sigma, self.max_sigma)
            return transforms.functional.gaussian_blur(img, kernel_size, sigma)
        else:
            return img
        
def jioayan(image):
    if np.random.random() < 0.5:
        image1 = np.array(image)
        # 添加椒盐噪声
        salt_vs_pepper_ratio = np.random.uniform(0.2, 0.4)
        amount = np.random.uniform(0.002, 0.006)
        num_salt = np.ceil(amount * image1.size * salt_vs_pepper_ratio)
        num_pepper = np.ceil(amount * image1.size * (1.0 - salt_vs_pepper_ratio))

        # 在随机位置生成椒盐噪声
        coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in image1.shape]
        coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in image1.shape]
        # image1[coords_salt] = 255
        image1[coords_salt[0], coords_salt[1], :] = 255
        image1[coords_pepper[0], coords_pepper[1], :] = 0
        image = Image.fromarray(image1)
    return image

def pengzhang(image):

    # 生成一个0到2之间的随机数
    random_value = random.random() * 3

    if random_value < 1:  # 1/3的概率进行加法操作
        he = random.randint(1, 3)
        kernel = np.ones((he, he), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
    elif random_value < 2:  # 1/3的概率进行除法操作
        he = random.randint(1, 3)  # 生成一个1到10之间的随机整数作为除数
        kernel = np.ones((he,he),np.uint8)
        image = cv2.dilate(image,kernel,iterations = 1)
    return image

class Mydata(Dataset):
    def __init__(self, dataFileName,transform=None):
        super(Mydata, self).__init__()
        # with open('Validation_train.json', 'r',encoding='utf8') as f:
        with open(dataFileName, 'r', encoding='utf8') as f:
            images = json.load(f)
            labels = images
        self.images, self.labels = images, labels
        self.transform = transform

    def __getitem__(self, item):
        # 读取图片
        image = Image.open(self.images[item]['path'].replace('\\','/'))
        # 转换
        if image.mode == 'L':
            image = image.convert('RGB')
        image_width, image_height = image.size
        if image_width > image_height:
            x = 72
            y = round(image_height / image_width * 72)
        # x, y = 72,72
        else:
            y = 72
            x = round(image_width / image_height * 72)
        sizey, sizex = 129, 129
        if y < 128:
            while sizey > 128 or sizey < 16:
                sizey = round(random.gauss(y, 30))
        if x < 128:
            while sizex > 128 or sizex < 16:
                sizex = round(random.gauss(x, 30))
        dx = 128 - sizex  # 差值
        dy = 128 - sizey
        if dx > 0:
            xl = -1
            while xl > dx or xl < 0:
                xl = round(dx / 2)
                xl = round(random.gauss(xl, 10))
        else:
            xl = 0
        if dy > 0:
            yl = -1
            while yl > dy or yl < 0:
                yl = round(dy / 2)
                yl = round(random.gauss(yl, 10))
        else:
            yl = 0
        yr = dy - yl
        xr = dx - xl
        image = jioayan(image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = pengzhang(image)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        random_gaussian_blur = RandomGaussianBlur()
        image = random_gaussian_blur(image)
        train_transform = transforms.Compose([
            transforms.Resize((sizey,sizex)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad([xl, yl, xr, yr], fill=(255, 255, 255), padding_mode='constant'),
            transforms.RandomRotation(degrees=(-15, 15), center=(round(64), round(64)), fill=(255, 255, 255)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.85233593, 0.85246795, 0.8517555], [0.31232414, 0.3122127, 0.31273854])])
        image = train_transform(image)
        label = torch.from_numpy(np.array(self.images[item]['label']))
        return image, label

    def __len__(self):
        return len(self.images)
# class Dataset:
#     def __init__(self,data,image_size,batch_size):
#         self.data = data
#         self.image_size = image_size
#         self.batch_size = batch_size


#     def load_data(self, path, kind='train'):
#         """Load Oracle-MNIST data from `path`"""
#         labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
#         images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

#         with gzip.open(labels_path, 'rb') as lbpath:
#             labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

#         with gzip.open(images_path, 'rb') as imgpath:
#             images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), self.image_size, self.image_size)

#         print('The size of %s set: %d'%(kind, len(labels)))

#         return images, labels


#     def preprocessing(self):
#         mean, std = (0.5,), (0.5,)
#         transform = transforms.Compose([transforms.ToTensor(),
#                                         transforms.Normalize(mean, std)
#                                         ])

#         x_train, y_train = self.load_data(f'{self.data}', kind='train')
#         x_test, y_test = self.load_data(f'{self.data}', kind='t10k')

#         x_train = x_train.reshape(-1, self.image_size, self.image_size, 1)
#         x_test = x_test.reshape(-1, self.image_size, self.image_size, 1)
#         x_train_tensor = torch.stack([transform(image.squeeze()) for image in x_train])
#         x_test_tensor = torch.stack([transform(image.squeeze()) for image in x_test])

#         train_dataset = TensorDataset(x_train_tensor, torch.tensor(y_train))
#         test_dataset = TensorDataset(x_test_tensor, torch.tensor(y_test))


#         trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
#         testloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

#         return trainloader,testloader