import torchvision
from torch import nn
from torch.nn import init
from models.utils import pooling
import torch
import torch.nn.functional as F
import random

class ResNet50(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1) 
        self.base = nn.Sequential(*list(resnet50.children())[:-2])

        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))

        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        
    def forward(self, x):
        x = self.base(x)
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)
        f = self.bn(x)

        return f

class VAE(nn.Module):
    def __init__(self,num_classes):
        super(VAE, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(4096, 1024)
        self.fc21 = nn.Linear(1024, self.num_classes)
        self.fc22 = nn.Linear(1024, self.num_classes)
        self.fc3 = nn.Linear(self.num_classes, 1024)
        self.fc4 = nn.Linear(1024, 4096)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 4096))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar

class DomainAug(nn.Module):
    def __init__(self, in_feat, class_nums):
        super(DomainAug, self).__init__()
        self.in_feat = in_feat
        self.num_classes = class_nums

        self.vae1 = VAE(num_classes = self.num_classes)
        self.vae2 = VAE(num_classes = self.num_classes)
        self.vae3 = VAE(num_classes = self.num_classes)
        self.vae4 = VAE(num_classes = self.num_classes)

    def forward(self, features):
        r1, z1, m1, l1 = self.vae1(features)
        r2, z2, m2, l2 = self.vae2(features)
        r3, z3, m3, l3 = self.vae3(features)
        r4, z4, m4, l4 = self.vae4(features)

        latent_features = torch.stack((z1,z2,z3,z4),dim=0) 
        recon_features = torch.stack((r1,r2,r3,r4),dim=0)
        # mu = torch.stack((m1,m2,m3,m4),dim=0)
        # var = torch.stack((l1,l2,l3,l4),dim=0)

        return latent_features, recon_features

class FeatAug(nn.Module):

    def __init__(self, probability=None):
        super(FeatAug, self).__init__()       
        self.probability = probability
        
    def forward(self, feats, aug_feats, targets): #
        
        N = feats.size()[0]
    
    #Inner sample Augmentation
        inner_feat = torch.zeros_like(feats)
        aug_feat = aug_feats[random.randint(0,3)]
        for i in range(N):                   
            f, _ = self.exchange_elements(feats[i],aug_feat[i])
            inner_feat[i] = f

            
    #Inter sample Augmentation 
        inter_feat = torch.zeros_like(feats)
        for i in range(0,N,2):
            if targets[i] == targets[i+1]:
                f1, f2 = self.exchange_elements(feats[i],feats[i+1])
                inter_feat[i] = f1
                inter_feat[i+1] = f2              

        return  F.normalize(inter_feat) , F.normalize(inner_feat) 
    

    def exchange_elements(self, f1,f2):

        D = len(f1)
        nums = torch.ones(D).cuda()
        nums[:int(self.probability*D)] = 0
        random.shuffle(nums)
           
        temp1 = torch.where(nums==0,f1,f2)
        temp2 = torch.where(nums==1,f1,f2)        

        return temp1, temp2