from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
import torch.utils.data.distributed
import argparse
import torch.utils.data
import shutil
from os import errno
from model import Net

parser = argparse.ArgumentParser(description='PYtorch criminal Training')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpuids', default=[0,1,2], nargs='+')
parser.add_argument('--data', default='dataset', metavar='DIR')
parser.add_argument('--workers', default=4, type= int, metavar='N')
parser.add_argument('--evalutate', dest='evaluate', action='store_true')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--lr', default=0.001)
opt = parser.parse_args()
print(opt)

opt.gpuids = list(map(int,opt.gpuids))


use_cuda = opt.cuda
if use_cuda and not torch.cuda.is_available():
    raise Exception("No Gpu found")


def main():
    model = Net()
    best_acc1 = 0
    if use_cuda:
        torch.cuda.set_device(opt.gpuids[0])
        model = nn.DataParallel(model, device_ids=opt.gpuids, output_device=opt.gpuids[0]).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(),lr = 0.0001)

    traindir = os.path.join(opt.data, 'train')
    valdir = os.path.join(opt.data, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32,
        shuffle=(train_sampler is None),
        num_workers=opt.workers,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=32, shuffle=False,
        num_workers=opt.workers,
        pin_memory=True)

    if opt.evaluate:
        validate(val_loader,model,criterion)
        return
    for epoch in range(0, opt.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch)

        losses = validate(val_loader, model, criterion)
        # remember best acc@1 and save checkpoint
        is_best = losses.val > best_acc1
        best_acc1 = min(losses.val, best_acc1)
        if(epoch%10 == 0):
            save_checkpoint(model, epoch)





def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()


    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #target = target.type(torch.FloatTensor)


        if use_cuda:
            model = model.cuda()
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input)


        #target = target.unsqueeze(1)        #targetsize 하고 put size하고 맞추기 위해서 이 함수를 씀.

        #torch.argmax(output,1)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            #target = target.type(torch.FloatTensor)

            if use_cuda:
                input = input.cuda()
                target = target.cuda()
                model = model.cuda()

            #target = target.unsqueeze(1)
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                i, len(val_loader), batch_time = batch_time, loss = losses))

        return losses

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opt.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(model, epoch):
    filename = 'model/model_epoch_{}.pth'.format(epoch)
    try:
        if not(os.path.isdir('model')):
            os.makedirs(os.path.join('model'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise
    torch.save(model, filename)
    print("checkpoint saved to {}".format(filename))



if __name__ == '__main__':
    main()