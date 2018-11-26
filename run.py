from __future__ import print_function
from os.path import join
import argparse
import torch
import os
import math
import torch.utils.data
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
from torchvision import datasets, models, transforms


parser = argparse.ArgumentParser(description='Pytorch crime classification')
parser.add_argument('--input_dir',default= "dataset/run", type=str)
parser.add_argument('--model', default="model_epoch_10.pth", type=str)
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)

model_name = join("model", opt.model)
model = torch.load(model_name)
rightcount = 0
wrongcount = 0


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(opt.input_dir, transform,)
    ,batch_size= 1
    ,shuffle= False
)

for i, (input, target) in enumerate(test_loader):
    input, target = Variable(input, volatile= False), Variable(target,volatile= False)
    if opt.cuda:
        model = model.cuda()
        input = input.cuda()


    out = model(input)
    out = out.cpu()
    out = torch.argmax(out,1)
    if(target == out):
        rightcount = rightcount + 1
    else:
        wrongcount = wrongcount + 1

print("accuracy : {:.4f}".format(rightcount/(wrongcount+rightcount)))

