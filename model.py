import torch.nn as nn
import torchvision.models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(1000,3)

        self.net = torchvision.models.alexnet(pretrained=True)
        #for param in self.net.parameters():
         #   param.requires_grad = False

    def forward(self, x):
        x1 = self.net(x)
        y = self.layer1(x1)
        return y

