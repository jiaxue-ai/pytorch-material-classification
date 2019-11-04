import torch
import torch.nn as nn
from torch.autograd import Variable


class DAIN(nn.Module):
    def __init__(self, nclass, model1, model2):
        super(DAIN, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.fc = nn.Linear(1280*2, nclass)

    def forward(self, img, diff_img):
        # pre-trained ResNet feature
        img_f = self.model1.features(img)
        img_f = img_f.mean(3).mean(2)

        # differential angular feature
        diff_img_f = self.model2.features(diff_img)
        diff_img_f = diff_img_f.mean(3).mean(2)

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
