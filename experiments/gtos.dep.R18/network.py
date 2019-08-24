import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Variable

import encoding
import torchvision.models as resnet

class DEPNet(nn.Module):
    def __init__(self, nclass, backbone='resnet18'):
        super(DEPNet, self).__init__()
        self.backbone = backbone
        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=True)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=True)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=True)
        elif backbone == 'resnet18':
            self.pretrained = resnet.resnet18(pretrained=True)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        n_codes = 8
        self.head = nn.Sequential(
            #nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            #nn.ReLU(inplace=True),
            encoding.nn.Encoding(D=512,K=n_codes),
            encoding.nn.View(-1, 512*n_codes),
            encoding.nn.Normalize(),
            nn.Linear(512*n_codes, 64),
            #nn.BatchNorm1d(64),
        )
        self.pool = nn.Sequential(
            nn.AvgPool2d(7),
            encoding.nn.View(-1, 512),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
        )
        self.fc = nn.Sequential(
            encoding.nn.Normalize(),
            nn.Linear(64*64, 128),
            encoding.nn.Normalize(),
            nn.Linear(128, nclass))
        # self.head = nn.Sequential(
        #     nn.Conv2d(512, 128, 1),
        #     encoding.nn.GramMatrix(),
        #     encoding.nn.View(-1, 128*128),
        #     nn.Linear(128*128, nclass)
        #     )

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            var_input = x 
            while not isinstance(var_input, Variable):
                var_input = var_input[0]
            _, _, h, w = var_input.size()
        else:
            raise RuntimeError('unknown input type: ', type(x))


        # pre-trained ResNet feature
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)

        # DEP head
        x1 = self.head(x)
        x2 = self.pool(x)
        x1 = x1.unsqueeze(1).expand(x1.size(0),x2.size(1),x1.size(-1))
        x = x1*x2.unsqueeze(-1)
        x=x.view(-1,x1.size(-1)*x2.size(1))
        x = self.fc(x)

        return x


def test():
    net = Net(nclass=23).cuda()
    print(net)
    x = Variable(torch.randn(1,3,224,224)).cuda()
    y = net(x)
    print(y)
    params = net.parameters()
    sum = 0
    for param in params:
        sum += param.nelement()
    print('Total params:', sum)


if __name__ == "__main__":
    test()
