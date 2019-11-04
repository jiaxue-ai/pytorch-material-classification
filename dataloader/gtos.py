import os
import os.path
import random

import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms


def find_classes(classdir):
    classes = []
    class_to_idx = {}
    with open(classdir, 'r') as f:
        for line in f:
            label, name = line.split(' ')
            classes.append(name)
            class_to_idx[name] = int(label) - 1
    return classes, class_to_idx


def make_dataset(txtname, datadir):
    rgbimages = []
    diffimages = []
    labels = []
    with open(txtname, "r") as lines:
        for line in lines:
            name, label = line.split(' ')
            name = name.split('/')[-1]
            for filename in os.listdir(os.path.join(datadir, 'diff_imgs', name)):
                _rgbimg = os.path.join(datadir, 'color_imgs', name, filename)
                _diffimg = os.path.join(datadir, 'diff_imgs', name, filename)
                assert os.path.isfile(_rgbimg)
                rgbimages.append(_rgbimg)
                diffimages.append(_diffimg)
                labels.append(int(label)-1) 

    return rgbimages, diffimages, labels


class GTOSDataloder(data.Dataset):
    def __init__(self, config, train=True, transform=None):
        classes, class_to_idx = find_classes(os.path.join(config.dataset_path, 'gtos_splits/classInd.txt'))
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.train = train
        self.transform = transform
        self.rgbnormalize = transforms.Normalize(mean=[0.447, 0.388, 0.340],
                                         std=[0.216, 0.204, 0.197])
        self.diffnormalize = transforms.Normalize(mean=[0.495, 0.495, 0.495],
                                         std=[0.07, 0.07, 0.07])
        
        if train:
            filename = os.path.join(config.dataset_path, 'gtos_splits/trainlist0'+ config.split +'.txt')
        else:
            filename = os.path.join(config.dataset_path, 'gtos_splits/testlist0'+ config.split +'.txt')

        self.rgbimages, self.diffimages, self.labels = make_dataset(filename, config.dataset_path)
        assert (len(self.rgbimages) == len(self.labels))

    def train_transform(self, _rgbimg, _diffimg):
        # sizes = [(224,224),(246,246),(268,268)]

        resize = transforms.Resize(size=(256, 256))
        # resize = transforms.Resize(size=sizes[random.randint(0,2)])
        _rgbimg, _diffimg = resize(_rgbimg), resize(_diffimg)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            _rgbimg, output_size=(224, 224))
        _rgbimg = TF.crop(_rgbimg, i, j, h, w)
        _diffimg = TF.crop(_diffimg, i, j, h, w)

        # Random horizontal and vertical flip
        if random.random() > 0.5:
            _rgbimg = TF.hflip(_rgbimg)
            _diffimg = TF.hflip(_diffimg)
        if random.random() > 0.5:
            _rgbimg = TF.vflip(_rgbimg)
            _diffimg = TF.vflip(_diffimg)

        # To tensor
        _rgbimg = TF.to_tensor(_rgbimg)
        _diffimg = TF.to_tensor(_diffimg)

        # Normalize
        _rgbimg = self.rgbnormalize(_rgbimg)
        _diffimg = self.diffnormalize(_diffimg)

        return _rgbimg, _diffimg

    def test_transform(self, _rgbimg, _diffimg):

        rgbtransform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.rgbnormalize,
        ])

        difftransform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.diffnormalize,
        ])

        _rgbimg = rgbtransform_test(_rgbimg)
        _diffimg = difftransform_test(_diffimg)

        return _rgbimg, _diffimg


    def __getitem__(self, index):
        _rgbimg = Image.open(self.rgbimages[index]).convert('RGB')
        _diffimg = Image.open(self.diffimages[index]).convert('RGB')
        _label = self.labels[index]
        if self.transform is not None:
            if self.train:
                _rgbimg, _diffimg = self.train_transform(_rgbimg, _diffimg)
            else:
                _rgbimg, _diffimg = self.test_transform(_rgbimg, _diffimg)


        return _rgbimg, _diffimg, _label

    def __len__(self):
        return len(self.rgbimages)
        # return 10000


class Dataloder():
    def __init__(self, config):

        trainset = GTOSDataloder(config, train=True, transform=True)
        testset = GTOSDataloder(config, train=False, transform=True)
    
        kwargs = {'num_workers': 0, 'pin_memory': True} if config.cuda else {}
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=
            config.batch_size, shuffle=True, **kwargs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=
            config.batch_size, shuffle=False, **kwargs)
        self.classes = trainset.classes
        self.trainloader = trainloader 
        self.testloader = testloader
    
    def getloader(self):
        return self.classes, self.trainloader, self.testloader



if __name__ == "__main__":
    trainset = GTOSDataloder(None, train=True)
    testset = GTOSDataloder(None, train=False)
    print(len(trainset.classes))
    print(len(testset))
