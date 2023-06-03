from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# ----------TODO------------
# Implement the PointNet 
# ----------TODO------------

# class PointNetfeat(nn.Module):
#     def __init__(self, global_feat = True, d=1024):
#         super(PointNetfeat, self).__init__()

#         self.d = d

#     def forward(self, x):
        
#         if self.global_feat:


#         else:

#             return 

class PointNetCls1024D(nn.Module):
    def __init__(self, k=2 ):
        super(PointNetCls1024D, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.mlp4 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.mlp5 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.mlp6 = nn.Linear(256, k)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        tmp = x.transpose(1, 2)
        x, _ = torch.max(x, dim=2)
        x = self.mlp4(x)
        x = self.mlp5(x)
        x = self.mlp6(x)

        return F.log_softmax(x, dim=1), tmp#, vis_feature # vis_feature only for visualization, your can use other ways to obtain the vis_feature


class PointNetCls256D(nn.Module):
    def __init__(self, k=2 ):
        super(PointNetCls256D, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.mlp4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp5 = nn.Linear(128, k)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x, _ = torch.max(x, dim=2)
        x = self.mlp4(x)
        x = self.mlp5(x)

        return F.log_softmax(x, dim=1), None





class PointNetSeg(nn.Module):
    def __init__(self, k = 2):
        super(PointNetSeg, self).__init__()
        self.k = k
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.mlp4 = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.mlp5 = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.mlp6 = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp7 = nn.Linear(128, k)


    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[1]
        x = x.transpose(1, 2)
        t = self.mlp1(x)
        x = self.mlp2(t)
        x = self.mlp3(x)
        x, _ = torch.max(x, dim=2, keepdim=True)
        x = torch.broadcast_to(x, (batchsize, 1024, n_pts))
        x = torch.cat((t, x), 1)
        x = self.mlp4(x)
        x = self.mlp5(x)
        x = self.mlp6(x)
        x = x.transpose(1, 2)
        x = self.mlp7(x)

        return F.log_softmax(x, dim=2)

