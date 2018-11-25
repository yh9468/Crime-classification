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
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)

model_name = join("model", opt.model)
model = torch.load(model_name)



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

rightcount = 0
wrongcount = 0

for i, (input, target) in enumerate(test_loader):

    if opt.cuda:
        model = model.cuda()
        input = input.cuda()


    out = model(input)
    out = out.cpu()
    out = torch.argmax(out,1)
    if(out == target):
        rightcount = rightcount+1
    else:
        wrongcount = wrongcount+1

    print("Accuracy : {.4f}".format(rightcount/(wrongcount+rightcount)))

