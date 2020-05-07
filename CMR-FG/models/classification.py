import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
from torchvision import models
from utils.config import cfg



class CLASSIFIER(nn.Module):
    def __init__(self):
        super(CLASSIFIER,self).__init__()
        self.L1 = nn.Linear(cfg.IMGF.embedding_dim,cfg.DATASET_TRAIN_CLSS_NUM)
        nn.init.xavier_uniform(self.L1.weight.data)
    def forward(self, input):
        x = self.L1(input)
        return x

class DISCRIMINATOR(nn.Module):
    def __init__(self):
        super(DISCRIMINATOR,self).__init__()
        self.L1=nn.Linear(1024,1)
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform(self.L1.weight.data)
    def forward(self,input):
        x = self.L1(input)
        x = self.sigmoid(x)        
        return x