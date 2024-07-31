# encoding: utf-8

import torch
from .triplet_loss import TripletLoss
import torch.nn.functional as F

def domain_recon_loss(feat_da, x):
    N = len(feat_da)
    loss = 0
    for i in range(N):
        recon_x = feat_da[i]
        loss = loss + F.mse_loss(recon_x, x.view(-1, recon_x.size(1)))
    return loss / N

def domain_tri_loss(feat_da, targets):
    triplet_loss = TripletLoss()
    N = len(feat_da)
    loss = 0
    for i in range(N):
        loss = loss + triplet_loss(feat_da[i], targets)
    return loss

def domain_align_loss(student, teacher):
    N = len(teacher)
    loss = 0
    for i in range(N):
        t = teacher[i]
        s = student[i]
        loss = loss + kl_distance(s,t) #
    return loss / N

def feat_aug_loss(feat_fa, targets):
    triplet_loss = TripletLoss()
    loss = triplet_loss(feat_fa, targets)
    return loss
    
def kl_distance(x,y):
    logp_x = F.log_softmax(x, dim=-1)
    p_y = F.softmax(y, dim=-1)
    kl_sum = F.kl_div(logp_x, p_y, reduction='batchmean')
    return kl_sum