# Author: David Harwath
import torch
import torch.nn as nn
import torch.nn.functional as F

        
class Davenet(nn.Module):
    def __init__(self):
        super(Davenet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(40,5), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(64, 512, kernel_size=(1,25), stride=(1,1), padding=(0,12))
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=(1,25), stride=(1,1), padding=(0,12))
        
        self.pool = nn.MaxPool2d(kernel_size=(1,4), stride=(1,2),padding=(0,1))
        

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # x = self.batchnorm1(x)
        x = self.conv1(x)
        x = F.relu(self.pool(x))
        x = self.conv2(x)
        x = F.relu(self.pool(x))
        x = F.relu(self.conv3(x))
        x = x.mean(2).mean(2)
        
        x = F.normalize(x, p = 2, dim = 1)
        

        return x