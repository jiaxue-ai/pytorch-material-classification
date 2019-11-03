import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Variable

import encoding

class DEPNet(nn.Module):
    def __init__(self, nclass, backbone):
        super(DEPNet, self).__init__()
        
        self.pretrained = backbone
        self.fc = nn.Linear(512, nclass)

    def forward(self, x):
        # pre-trained ResNet feature
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)
        x = self.pretrained.avgpool(x)

        # finetune head
        x = torch.flatten(x, 1)
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
