import matplotlib.pyplot as plt
import cv2
import os 
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
import torch
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
    def __init__(self, root, transform=None):
        self.dataset_path = root
        self.transform = transform
        self.image_dir = os.path.join(self.dataset_path, 'img')
        self.mask_dir = os.path.join(self.dataset_path, 'mask')
        self.image_list = [f for f in os.listdir(self.image_dir) 
                           if os.path.isfile(os.path.join(self.image_dir, f))]
        self.mask_list = [f for f in os.listdir(self.mask_dir)
                          if os.path.isfile(os.path.join(self.mask_dir, f))]
        self.grayscale = transforms.Grayscale()
        
    
    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = os.path.join(self.image_dir, image_name)
        mask_file = os.path.join(self.mask_dir, image_name) if os.path.isfile(os.path.join(self.mask_dir, image_name)) else None

        img = Image.open(image_file)
        mask = self.grayscale(Image.open(mask_file))
        
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        return img, mask
    
    def __len__(self):
        return len(self.image_list) 
        

def trainloader(colab=False, batch_size=1, transform=transforms.ToTensor()):
    path = 'datasets/256-256-train/'
    if colab:
        path = '../drive/My Drive/Colab Notebooks/DATA/256-256-train/'

    train_data = RemoteImageDataset(root=path, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return train_loader


def validloader(colab=False, batch_size=1, transform=transforms.ToTensor()):
    path = 'datasets/256-256-val'
    if colab:
        path = '../drive/My Drive/Colab Notebooks/DATA/256-256-val/'

    val_data = RemoteImageDataset(root=path, transform=transform) 
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return val_loader


def testloader(colab=False, batch_size=1, transform=transforms.ToTensor()):
    path = 'datasets/256-256-test'
    if colab:
        path = '../drive/My Drive/Colab Notebooks/DATA/256-256-test/'

    test_data = RemoteImageDataset(root=path, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return test_loader


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # dataset = RemoteImageDataset(root='datasets/256-256-train/', transform=transform)
    # print(len(dataset))
    dataset = validloader()
    print(len(dataset))
    # dataset = iter(dataset)
    # dataset = dataset.next()
    # img = dataset[0]
    # mask = dataset[1]
    # plt.imshow(img.reshape(3,256,256).permute(1,2,0))
    # plt.show()
