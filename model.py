import torch.nn as nn
import torch
import torchvision.models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model_conv = torchvision.models.resnet50(pretrained=True)
        self.numftrs = self.model_conv.fc.in_features
        self.model_conv.fc = nn.Linear(self.numftrs,3)


        #for param in self.net.parameters():
         #   param.requires_grad = False

    def forward(self, x):
        y = self.model_conv(x)
        return y

