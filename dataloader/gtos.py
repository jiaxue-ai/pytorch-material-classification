import torch
import torch.utils.data as data
import torchvision
from torchvision import datasets, models, transforms

from PIL import Image
import os
import os.path

_imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


class Dataloder():
    def __init__(self, config):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4,0.4,0.4),
            transforms.ToTensor(),
            Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        data_dir = 'dataset/gtos-mobile'
        trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform_train)
        testset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform_test)


        kwargs = {'num_workers': 8, 'pin_memory': True} if config.cuda else {}
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=
            config.batch_size, shuffle=True, **kwargs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=
            config.batch_size, shuffle=False, **kwargs)


        self.trainloader = trainloader 
        self.testloader = testloader
        self.classes = trainset.classes
    
    def getloader(self):
        return self.classes, self.trainloader, self.testloader


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


if __name__ == "__main__":
    data_dir = 'dataset/gtos-mobile'
    trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'))
    testset = datasets.ImageFolder(os.path.join(data_dir, 'test'))
    print(trainset.classes)
    print(len(testset))
