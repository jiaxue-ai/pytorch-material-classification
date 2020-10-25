import torch
import torch.nn as nn
from torch.autograd import Variable


class TEAN(nn.Module):
    def __init__(self, nclass, model1, model2):
        super(TEAN, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model1.classifier[1] = nn.Linear(128+128, num_classes)

        self.head = nn.Sequential(
            encoding.nn.Encoding(D=1280,K=n_codes),
            encoding.nn.View(-1, 1280*n_codes),
            encoding.nn.Normalize(),
            #nn.ReLU(inplace=True),
            nn.Linear(1280*n_codes, 64),
            nn.BatchNorm1d(64),
        )
        self.pool = nn.Sequential(
            nn.AvgPool2d(7),
            encoding.nn.View(-1, 1280),
            nn.Linear(1280, 64),
            nn.BatchNorm1d(64),
        )
        self.fc = nn.Sequential(
            encoding.nn.Normalize(),
            #nn.ReLU(inplace=True),
            nn.Linear(64*64, 128),
            encoding.nn.Normalize(),
            )
        self.pool2 = nn.Sequential(
            nn.AvgPool2d(7),
            encoding.nn.View(-1, 1280),
            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
        )

    def forward(self, img, diff_img):
        # pre-trained ResNet feature
        img_f = self.model1.features(img)

        # differential angular feature
        diff_img_f = self.model2.features(diff_img)
        diff_img_f = img_f + diff_img_f
        diff_img_f = self.pool2(diff_img_f)

        # dep feature
        x1 = self.head(img_fea)
        x2 = self.pool(img_fea)
        x1 = x1.unsqueeze(1).expand(x1.size(0),x2.size(1),x1.size(-1))
        x = x1*x2.unsqueeze(-1)
        enc_fea = x.view(-1,x1.size(-1)*x2.size(1))
        enc_fea = self.fc(enc_fea)

        # TEAN head
        out = torch.cat((enc_fea, ang_fea), dim=1)
        out = self.model1.classifier(out)

        return out


def test():
    net = TEAN(nclass=23).cuda()
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
