import matplotlib.pyplot as plt
import cv2
import os 
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.transforms.functional as tfunc
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
import torch
import random
from PIL import Image


def datacounter(root='pkm/train/'):
    classes = ['bulbasaur', 'charmander', 'jigglypuff', 'magikarp', 'mudkip', 'pikachu', 'psyduck', 'snorlax', 'squirtle']
    X, y = [], []
    for index, clsname in enumerate(classes):
        imgpath = os.path.join(root, clsname)
        for path in os.listdir(imgpath):
            imgfile = os.path.join(imgpath, path)
            X.append(imgfile)
            y.append(index)
    return X, y


def dataplot(y_train, y_test):    
    f, axes = plt.subplots(1, 2, figsize=(18, 6))
    ax1, ax2 = axes
    sns.countplot(y_train, ax=ax1)
    ax1.set_xlabel('train')
    ax1.set_ylabel('number')
    ax1.set_ylim([0, 200])
    sns.countplot(y_test, ax=ax2)
    ax2.set_xlabel('test')
    ax2.set_ylabel('number')
    ax2.set_ylim([0, 200])
    plt.show()


class RemoteImageDataset(Dataset):
    def __init__(self, root, augment=False):
        self.dataset_path = root
        self.augment = augment
        self.image_dir = os.path.join(self.dataset_path, 'img')
        self.mask_dir = os.path.join(self.dataset_path, 'mask')
        self.image_list = [f for f in os.listdir(self.image_dir) 
                           if os.path.isfile(os.path.join(self.image_dir, f))]
        self.mask_list = [f for f in os.listdir(self.mask_dir)
                          if os.path.isfile(os.path.join(self.mask_dir, f))]
        self.grayscale = transforms.Grayscale()
        self.totensor = transforms.ToTensor()
    
    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = os.path.join(self.image_dir, image_name)
        mask_file = os.path.join(self.mask_dir, image_name) if os.path.isfile(os.path.join(self.mask_dir, image_name)) else None

        img = Image.open(image_file)
        mask = Image.open(mask_file)
        mask = self.grayscale(mask)
        
        img = self.totensor(img)
        mask = self.totensor(mask)
        
        if self.augment:
            return augmentor(img, mask)

        return img, mask
    
    def __len__(self):
        return len(self.image_list)


def augmentor(img, mask):
    img = tfunc.to_pil_image(img)
    mask = tfunc.to_pil_image(mask)

    if random.random() >= 0.5:
        img = tfunc.hflip(img)
        mask = tfunc.hflip(mask)

    if random.random() >= 0.5:
        img = tfunc.vflip(img)
        mask = tfunc.vflip(mask)

    if random.random() >= 0.5:
        img = tfunc.rotate(img, 90)
        mask = tfunc.rotate(mask, 90)

    img = tfunc.to_tensor(img)
    mask = tfunc.to_tensor(mask)

    return img, mask

def trainloader(colab=False, batch_size=1, augment=False):
    path = 'datasets/256-256-train/'
    if colab:
        path = '../drive/My Drive/Colab Notebooks/DATA/256-256-train/'

    train_data = RemoteImageDataset(root=path, augment=augment)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return train_loader


def validloader(colab=False, batch_size=1, augment=False):
    path = 'datasets/256-256-val'
    if colab:
        path = '../drive/My Drive/Colab Notebooks/DATA/256-256-val/'

    val_data = RemoteImageDataset(root=path, augment=augment) 
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return val_loader


def testloader(colab=False, batch_size=1, augment=False):
    path = 'datasets/256-256-test'
    if colab:
        path = '../drive/My Drive/Colab Notebooks/DATA/256-256-test/'

    test_data = RemoteImageDataset(root=path, augment=augment)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return test_loader


if __name__ == '__main__':

    data = trainloader(augment=False)
    for img, mask in iter(data):
        # print(img.reshape())
        img_, mask_ = augmentor(img.reshape(3,256,256), mask.reshape(3,256,256))
        img_ = img_.permute(1,2,0)
        mask_ = mask_.permute(1,2,0)
        img = img.permute(0,2,3,1).reshape(256,256,3)
        mask = mask.permute(0,2,3,1).reshape(256,256,3)

        fig, (ax1, ax2,a3,a4) = plt.subplots(1,4, figsize=(8,3))
        ax1.imshow(img)
        ax2.imshow(mask)
        a3.imshow(img_)
        a4.imshow(mask_)

        plt.show()
        break