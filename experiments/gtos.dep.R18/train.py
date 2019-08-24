from __future__ import print_function

import matplotlib.pyplot as plot

import torch
import torch.nn as nn
from config import config
from model.dep import DEPNet
from dataloader.gtos import Dataloder
import torch.optim as optim
from torch.autograd import Variable

from option import Options
from utils import *
import sys
import os

# global variable
best_pred = 100.0
errlist_train = []
errlist_val = []


def adjust_learning_rate(optimizer, config, epoch, best_pred):
    lr = config.lr * (0.1 ** ((epoch - 1) // config.lr_decay))
    if (epoch-1) % config.lr_decay == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    # init the config
    global best_pred, errlist_train, errlist_val
    config.cuda = not config.no_cuda and torch.cuda.is_available()
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed(config.seed)
    # init dataloader
    classes, train_loader, test_loader = Dataloder(config).getloader()

    # init the model
    model = Net(len(classes))
    print(model)
    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
     weight_decay=config.weight_decay)
    if config.cuda:
        model.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        model = torch.nn.DataParallel(model)

    # check point
    if config.resume is not None:
        if os.path.isfile(config.resume):
            print("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            config.start_epoch = checkpoint['epoch'] +1
            best_pred = checkpoint['best_pred']
            errlist_train = checkpoint['errlist_train']
            errlist_val = checkpoint['errlist_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(config.resume, checkpoint['epoch']))
        else:
            print("=> no resume checkpoint found at '{}'".\
                format(config.resume))

    def train(epoch):
        model.train()
        global best_pred, errlist_train
        train_loss, correct, total = 0,0,0
        adjust_learning_rate(optimizer, config, epoch, best_pred)
        for batch_idx, (data, target) in enumerate(train_loader):

            #scheduler(optimizer, batch_idx, epoch, best_pred)
            if config.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            pred = output.data.max(1)[1] 
            correct += pred.eq(target.data).cpu().sum()
            total += target.size(0)
            err = 100-100.*correct/total
            progress_bar(batch_idx, len(train_loader), 
                'Loss: %.3f | Err: %.3f%% (%d/%d)' % \
                (train_loss/(batch_idx+1), 
                err, total-correct, total))
        errlist_train += [err]

    def test(epoch):
        model.eval()
        global best_pred, errlist_train, errlist_val
        test_loss, correct, total = 0,0,0
        is_best = False
        for batch_idx, (data, target) in enumerate(test_loader):
            if config.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).data[0]
            # get the index of the max log-probability
            pred = output.data.max(1)[1] 
            correct += pred.eq(target.data).cpu().sum()
            total += target.size(0)

            err = 100-100.*correct/total
            progress_bar(batch_idx, len(test_loader), 
                'Loss: %.3f | Err: %.3f%% (%d/%d)'% \
                (test_loss/(batch_idx+1), 
                err, total-correct, total))

        if config.eval:
            print('Error rate is %.3f'%err)
            return
        # save checkpoint
        errlist_val += [err]
        if err < best_pred:
            best_pred = err 
            is_best = True
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
            'errlist_train':errlist_train,
            'errlist_val':errlist_val,
            }, args=config, is_best=is_best)
            

    if config.eval:
        test(config.start_epoch)
        return

    for epoch in range(config.start_epoch, config.nepochs + 1):
        print('Epoch:', epoch)
        train(epoch)
        test(epoch)

if __name__ == "__main__":
    main()
