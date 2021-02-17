import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
from torchvision import models
from utils.config import cfg


class Resnet101(nn.Module):
    def __init__(self):
        super(Resnet101, self).__init__()
        model = models.resnet101(pretrained=True)       
        for param in model.parameters():
            param.requires_grad = False        
        self.define_module(model)       

    def define_module(self, model):
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool       
        self.fc = nn.Linear(2048,2048)        
        self.bnorm = nn.BatchNorm1d(2048)
        # self.self_att = self_AttentionModel(380,cfg.TRAIN.SMOOTH.IMGATT2)
        self.ca = ChannelAttention(in_planes=1024)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = nn.functional.interpolate(x,size=(244, 244), mode='bilinear', align_corners=False)    # (3, 244, 244)
        x = self.conv1(x)    # (64, 122, 122)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)     #(256, 61, 61)     
        x = self.layer2(x)     #(512, 31, 31)        
        x = self.layer3(x)        #(1024, 16, 16)    
        if cfg.image_attention:  
            x = self.ca(x) * x
            x = self.sa(x) * x  
        x = self.layer4(x)        #(2048, 8, 8) 
            
        x = self.avgpool(x)   

        x = x.view(x.shape[0],-1)       
        x = self.bnorm(x)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)            
        return x #,features,loc_feature
        
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)