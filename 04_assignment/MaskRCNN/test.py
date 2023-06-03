from tkinter import Label
import utils
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNN
from dataset import SingleShapeDataset, ShapeDataset
from utils import plot_save_output
import torch
import numpy as np
import torch.utils.data




# the outputs includes: 'boxes', 'labels', 'masks', 'scores'

def iou(A, B):
    i = max(min(A[2], B[2]) - max(A[0], B[0]) , 0) * max(min(A[3], B[3]) - max(A[1], B[1]) , 0)
    u = (A[2] - A[0]) * (A[3] - A[1]) + (B[2] - B[0]) * (B[3] - B[1]) - i
    return i / u

def IOU(A, B):
    I = torch.sum(torch.logical_and(A >= 0.5, B >= 0.5))
    U = torch.sum(torch.logical_or(A >= 0.5, B >= 0.5))
    return I / U

def compute_segmentation_ap(output_list, gt_labels_list, iou_threshold=0.5):
    ans = 0.0
    for label in range(1, 4, 1):
        l = []
        sco = []
        for i in range(len(output_list)):
            output = output_list[i]
            for j in range(output['labels'].shape[0]):
                if output['labels'][j] == label:
                    l.append([output['masks'][j][0], i])
                    sco.append(output['scores'][j])
        for i in range(len(sco)):
            for j in range(i+1, len(sco), 1):
                if sco[i] < sco[j]:
                    sco[i], sco[j] = sco[j], sco[i]
                    l[i], l[j] = l[j], l[i]
        called = []
        zong = 0
        for i in range(len(gt_labels_list)):
            called.append([])
            for j in range(len(gt_labels_list[i]['boxes'])):
                called[i].append(0)
                if gt_labels_list[i]['labels'][j] == label:
                    zong += 1
        f = []
        for i in range(11):
            f.append(0.0)
        call = 0
        dui = 0
        for i in range(len(l)):
            ok = 0
            pic = l[i][1]
            for j in range(len(gt_labels_list[pic]['boxes'])):
                if gt_labels_list[pic]['labels'][j] == label:
                    if IOU(l[i][0], gt_labels_list[pic]['masks'][j]) >= iou_threshold:
                        ok = 1
                        if called[pic][j] == 0:
                            called[pic][j] = 1
                            call += 1
            dui += ok
            precision = dui / (i+1)
            recall = call / zong
            for _ in range(11):
                if recall >= _ / 10:
                    f[_] = max(f[_], precision)
        print(f)
        ap = sum(f) / 11
        ans += ap / 3

    return ans



def compute_detection_ap(output_list, gt_labels_list, iou_threshold=0.5):
    ans = 0.0
    for label in range(1, 4, 1):
        l = []
        sco = []
        for i in range(len(output_list)):
            output = output_list[i]
            for j in range(output['labels'].shape[0]):
                if output['labels'][j] == label:
                    l.append([output['boxes'][j], i])
                    sco.append(output['scores'][j])
        for i in range(len(sco)):
            for j in range(i+1, len(sco), 1):
                if sco[i] < sco[j]:
                    sco[i], sco[j] = sco[j], sco[i]
                    l[i], l[j] = l[j], l[i]
        called = []
        zong = 0
        for i in range(len(gt_labels_list)):
            called.append([])
            for j in range(len(gt_labels_list[i]['boxes'])):
                called[i].append(0)
                if gt_labels_list[i]['labels'][j] == label:
                    zong += 1
        f = []
        for i in range(11):
            f.append(0.0)
        call = 0
        dui = 0
        for i in range(len(l)):
            ok = 0
            pic = l[i][1]
            for j in range(len(gt_labels_list[pic]['boxes'])):
                if gt_labels_list[pic]['labels'][j] == label:
                    if iou(l[i][0], gt_labels_list[pic]['boxes'][j]) >= iou_threshold:
                        ok = 1
                        if called[pic][j] == 0:
                            called[pic][j] = 1
                            call += 1
            dui += ok
            precision = dui / (i+1)
            recall = call / zong
            for _ in range(11):
                if recall >= _ / 10:
                    f[_] = max(f[_], precision)
        print(f)
        ap = sum(f) / 11
        ans += ap / 3

    return ans







dataset_test = ShapeDataset(10)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)
 

num_classes = 4
 
# get the model using the helper function
model = utils.get_instance_segmentation_model(num_classes).double()

device = torch.device('cpu')


# replace the 'cpu' to 'cuda' if you have a gpu
model.load_state_dict(torch.load(r'results/maskrcnn_75.pth',map_location='cpu'))
#model.load_state_dict(torch.load(r'../intro2cv_maskrcnn_pretrained.pth',map_location='cpu'))



model.eval()
path = "../results/MaskRCNN/" 
# save visual results
for i in range(10):
    imgs, labels = dataset_test[i]
    output = model([imgs])
    plot_save_output(path+str(i)+"_result.png", imgs, output[0])

# compute AP
gt_labels_list = []
output_label_list = []
with torch.no_grad():
    for i in range(10):
        print(i)
        imgs, labels = dataset_test[i]
        gt_labels_list.append(labels)
        output = model([imgs])
        output_label_list.append(output[0])

mAP_detection = compute_detection_ap(output_label_list, gt_labels_list)
mAP_segmentation = compute_segmentation_ap(output_label_list, gt_labels_list)


np.savetxt(path+"mAP.txt",np.asarray([mAP_detection, mAP_segmentation]))

