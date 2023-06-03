from __future__ import print_function
import os
import random
import torch
import torch.nn.parallel
import torch.utils.data
from dataset import ShapeNetClassficationDataset
from model import PointNetCls1024D
import numpy as np
from utils import write_points, setting
import cv2


    

if __name__ == '__main__':
    feat_dim = 1024
    opt = setting()
    blue = lambda x: '\033[94m' + x + '\033[0m'
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)


    test_dataset = ShapeNetClassficationDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        with_data_augmentation=False)

   

    testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=5,
            shuffle=True,
            num_workers=int(opt.workers))

    num_classes = len(test_dataset.classes)
    print('classes', num_classes)


    classifier = PointNetCls1024D(k=num_classes)

    # load weights:
    classifier.load_state_dict(torch.load(f"../exps/cls_{feat_dim}D/model.pth"))

    classifier.eval()

    for i,data in enumerate(testdataloader, 0):
        points, target = data
        target = target[:, 0]
        classifier = classifier.eval()
        pred, heat_feat = classifier(points)
        heat_feat = heat_feat.detach().numpy()
        heat_feat = np.max(heat_feat, 2)
        heat_feat = (heat_feat - np.min(heat_feat, axis=1, keepdims=True)) / (np.max(heat_feat, axis=1, keepdims=True) - np.min(heat_feat, axis=1, keepdims=True))
        #heat_feat = (heat_feat - np.min(heat_feat,axis=0))/(np.max(heat_feat,axis=0) - np.min(heat_feat,axis=0))
        color_heat_feat = cv2.applyColorMap((heat_feat*255).astype(np.uint8), cv2.COLORMAP_JET ) #BGR



        write_points( os.path.join(opt.resultf, str(0)+".ply"), points.numpy()[0,...], color_heat_feat[0,...], heat_feat[0,...])
        write_points( os.path.join(opt.resultf, str(1)+".ply"), points.numpy()[1,...], color_heat_feat[1,...], heat_feat[1,...])
        write_points( os.path.join(opt.resultf, str(2)+".ply"), points.numpy()[2,...], color_heat_feat[2,...], heat_feat[2,...])
        write_points( os.path.join(opt.resultf, str(3)+".ply"), points.numpy()[3,...], color_heat_feat[3,...], heat_feat[3,...])
        write_points( os.path.join(opt.resultf, str(4)+".ply"), points.numpy()[4,...], color_heat_feat[4,...], heat_feat[4,...])

        break
 

