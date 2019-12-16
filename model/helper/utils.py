import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import imgaug as ia
from imgaug import augmenters as iaa
from datetime import datetime, date
import os, glob, shutil
mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 5, 10


def calc_norm(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1147, shuffle=False, num_workers=4)
    pop_mean = []
    pop_std0 = []
    pop_std1 = []

    for i, data in enumerate(dataloader, 0):
        numpy_image = data['image'].numpy()
        
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
        
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)

    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)
    
    print(pop_mean) # [0.6855248  0.68901044 0.6142709 ] mean
    print(pop_std0) # [0.32218283 0.27970755 0.31340986] std0
    print(pop_std1) # [0.32218322 0.27970782 0.3134101 ] std1


def show_dataset(dataset, n=6):
    img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n))) for i in range(5)))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


class ImgAugmenter:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.4, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Sometimes(0.3, iaa.OneOf([iaa.Dropout(p=(0, 0.1)), iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])
        
    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class Logger(object):
    """ Simple training logger: saves to file and optionally prints to stdout """
    def __init__(self, logname):
        """
        Args:
            logname: name for log (e.g. 'Hopper-v1')
            now: unique sub-directory name (e.g. date/time string)
        """
        now = date.today().strftime('%d-%m-%Y_') + datetime.now().strftime('%H:%M:%S')
        path = os.path.join('log-files', logname, now)
        os.makedirs(path)
        folders = [('','*.py'), ('model', '*.py')]
        # put copy of all python files in log_dir
        for folder, query in folders:
            filenames = glob.glob(os.path.join(folder, query))
            path_ = os.path.join(path, folder)
            if not os.path.exists(path_): os.makedirs(path_)
            for filename in filenames:     # for reference
                shutil.copy(filename, path_)
        path = os.path.join(path, 'log.csv')

        self.write_header = True
        self.log_entry = {}
        self.f = open(path, 'w')
        self.writer = None  # DictWriter created with first call to write() method

    def write(self, display=True):
        """ Write 1 log entry to file, and optionally to stdout
        Log fields preceded by '_' will not be printed to stdout

        Args:
            display: boolean, print to stdout
        """
        if display:
            self.disp(self.log_entry)
        if self.write_header:
            fieldnames = [x for x in self.log_entry.keys()]
            self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
            self.writer.writeheader()
            self.write_header = False
        self.writer.writerow(self.log_entry)
        self.log_entry = {}

    @staticmethod
    def disp(log):
        """Print metrics to stdout"""
        log_keys = [k for k in log.keys()]
        log_keys.sort()
        # print('***** Episode {}, Mean R = {:.1f} *****'.format(log['_Episode'],
                                                            #    log['_MeanReward']))
        for key in log_keys:
            if key[0] != '_':  # don't display log items with leading '_'
                print('{:s}: {:.3g}'.format(key, log[key]))

    def log(self, items):
        """ Update fields in log (does not write to file, used to collect updates.

        Args:
            items: dictionary of items to update
        """
        self.log_entry.update(items)

    def close(self):
        """ Close log file - log cannot be written after this """
        self.f.close()


class EarlyStopping(object):
    def __init__(self, mode='min', delta=0, patience=10, percentage=False):
        self.mode = mode
        self.delta = delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False


    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False


    def _init_is_better(self, percentage):
        if self.mode not in ['min', 'max']:
            raise ValueError('mode ' + self.mode + ' is unknown!')
        if not percentage:
            if self.mode == 'min':
                self.is_better = lambda a, best: a < best - self.delta
            if self.mode == 'max':
                self.is_better = lambda a, best: a > best + self.delta
        else:
            if self.mode == 'min':
                self.is_better = lambda a, best: a < best - (best * self.delta / 100)
            if self.mode == 'max':
                self.is_better = lambda a, best: a > best + (best * self.delta / 100)


if __name__ == '__main__':
    from dataloader import dataloader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    import PIL

    transform = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            ImgAugTransform(),
            lambda x: PIL.Image.fromarray(x),
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(15),
            transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    }

    train_data = ImageFolder(root='../pkm/train/', transform=transform['train'])
    # show_dataset(train_data)
    data, labels = train_data[888]
    plt.imshow(data.permute(1,2,0))
    plt.show()