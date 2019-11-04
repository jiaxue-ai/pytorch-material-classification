import torch
import torch.nn as nn
from torch.autograd import Variable


class DAIN(nn.Module):
    def __init__(self, nclass, model1, model2):
        super(DAIN, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.fc = nn.Linear(512*2, nclass)

    def forward(self, img, diff_img):
        # pre-trained ResNet feature
        img_f = self.model1.conv1(img)
        img_f = self.model1.bn1(img_f)
        img_f = self.model1.relu(img_f)
        img_f = self.model1.maxpool(img_f)
        img_f = self.model1.layer1(img_f)
        img_f = self.model1.layer2(img_f)
        img_f = self.model1.layer3(img_f)
        img_f = self.model1.layer4(img_f)
        img_f = self.model1.avgpool(img_f)

        # differential angular feature
        diff_img_f = self.model2.conv1(diff_img)
        diff_img_f = self.model2.bn1(diff_img_f)
        diff_img_f = self.model2.relu(diff_img_f)
        diff_img_f = self.model2.maxpool(diff_img_f)
        diff_img_f = self.model2.layer1(diff_img_f)
        diff_img_f = self.model2.layer2(diff_img_f)
        diff_img_f = self.model2.layer3(diff_img_f)
        diff_img_f = self.model2.layer4(diff_img_f)
        diff_img_f = self.model2.avgpool(diff_img_f)

        # DAIN head
        img_f = torch.flatten(img_f, 1)
        diff_img_f = torch.flatten(diff_img_f, 1)
        diff_img_f = diff_img_f + img_f
        out = torch.cat((img_f, diff_img_f), dim=1)
        out = self.fc(out)

        return out


def test():
    net = DAIN(nclass=23).cuda()
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
