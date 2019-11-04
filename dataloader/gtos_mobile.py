import os
import os.path

import torch
from utils.data_aug import Lighting
from torchvision import datasets, transforms

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
            Lighting(0.1),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        trainset = datasets.ImageFolder(os.path.join(config.dataset_path, 'train'), transform_train)
        testset = datasets.ImageFolder(os.path.join(config.dataset_path, 'test'), transform_test)


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


if __name__ == "__main__":
    data_dir = 'dataset/gtos-mobile'
    trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'))
    testset = datasets.ImageFolder(os.path.join(data_dir, 'test'))
    print(trainset.classes)
    print(len(testset))
