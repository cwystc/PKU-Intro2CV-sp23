import os
from matplotlib import image
import numpy as np
import torch
import torch.utils.data
import random
import cv2
import math 
from utils import plot_save_dataset

def boxarea(a):
    return (a[2] - a[0]) * (a[3] - a[1])
def box_intersect(A, B):
    i = max(min(A[2], B[2]) - max(A[0], B[0]) , 0) * max(min(A[3], B[3]) - max(A[1], B[1]) , 0)
    #u = (A[2] - A[0]) * (A[3] - A[1]) + (B[2] - B[0]) * (B[3] - B[1]) - i
    return i

class SingleShapeDataset(torch.utils.data.Dataset):
    def __init__(self, size):

        self.w = 128
        self.h = 128
        self.size = size
        print("size",self.size)


    def _draw_shape(self, img, mask, shape_id):
        buffer = 20
        y = random.randint(buffer, self.h - buffer - 1)
        x = random.randint(buffer, self.w - buffer - 1)
        s = random.randint(buffer, self.h//4)
        color = tuple([random.randint(0, 255) for _ in range(3)])

        
        if shape_id == 1:
            cv2.rectangle(mask, (x-s, y-s), (x+s, y+s), 1, -1)
            cv2.rectangle(img, (x-s, y-s), (x+s, y+s), color, -1)

        elif shape_id == 2:
            cv2.circle(mask, (x, y), s, 1, -1)
            cv2.circle(img, (x, y), s, color, -1)

        elif shape_id == 3:
            points = np.array([[(x, y-s),
                            (x-s/math.sin(math.radians(60)), y+s),
                            (x+s/math.sin(math.radians(60)), y+s),
                            ]], dtype=np.int32)
            cv2.fillPoly(mask, points, 1)
            cv2.fillPoly(img, points, color)


    def __getitem__(self, idx):
        np.random.seed(idx)

        n_class = 1
        masks = np.zeros((n_class, self.h, self.w))
        img = np.zeros((self.h, self.w, 3))
        img[...,:] = np.asarray([random.randint(0, 255) for _ in range(3)])[None, None, :]


        obj_ids = np.zeros((n_class)) 

        shape_code = random.randint(1,3)
        self._draw_shape( img, masks[0, :], shape_code)
        obj_ids[0] = shape_code


        boxes = np.zeros((n_class,4))
        pos = np.where(masks[0])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes[0,:] = np.asarray([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        img = torch.tensor(img)
        img = img.permute(2,0,1)


        return img, target

    def __len__(self):
        return self.size




# ----------TODO------------
# Implement ShapeDataset.
# Refer to `SingleShapeDataset` for the shape parameters 
# ----------TODO------------



class ShapeDataset(torch.utils.data.Dataset):
    def __init__(self, size):

        self.w = 128
        self.h = 128
        self.size = size
        print("size",self.size)


    def gen_shape(self, img, mask, shape_id):
        buffer = 20
        y = random.randint(buffer, self.h - buffer - 1)
        x = random.randint(buffer, self.w - buffer - 1)
        s = random.randint(buffer, self.h//4)
        color = tuple([random.randint(0, 255) for _ in range(3)])

        
        if shape_id == 1:
            cv2.rectangle(mask, (x-s, y-s), (x+s, y+s), 1, -1)

        elif shape_id == 2:
            cv2.circle(mask, (x, y), s, 1, -1)

        elif shape_id == 3:
            points = np.array([[(x, y-s),
                            (x-s/math.sin(math.radians(60)), y+s),
                            (x+s/math.sin(math.radians(60)), y+s),
                            ]], dtype=np.int32)
            cv2.fillPoly(mask, points, 1)
        return x, y, s, color
        
    def draw_shape(self, x, y, s, color, img, mask, shape_id):

        
        if shape_id == 1:
            cv2.rectangle(img, (x-s, y-s), (x+s, y+s), color, -1)

        elif shape_id == 2:
            cv2.circle(img, (x, y), s, color, -1)

        elif shape_id == 3:
            points = np.array([[(x, y-s),
                            (x-s/math.sin(math.radians(60)), y+s),
                            (x+s/math.sin(math.radians(60)), y+s),
                            ]], dtype=np.int32)
            cv2.fillPoly(img, points, color)


    def __getitem__(self, idx):
        np.random.seed(idx)

        n_class = random.randint(1, 10)
        obj_ids = np.zeros((n_class))
        masks = np.zeros((n_class, self.h, self.w))
        img = np.zeros((self.h, self.w, 3))
        img[...,:] = np.asarray([random.randint(0, 255) for _ in range(3)])[None, None, :]
        image_id = torch.tensor([idx])
        boxes = np.zeros((n_class,4))

        X = []
        Y = []
        S = []
        COLOR = []
        for _ in range(n_class):
            shape_code = random.randint(1,3)

            x, y, s, color = self.gen_shape( img, masks[_, :], shape_code)
            X.append(x)
            Y.append(y)
            S.append(s)
            COLOR.append(color)
            obj_ids[_] = shape_code


            pos = np.where(masks[_])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes[_,:] = np.asarray([xmin, ymin, xmax, ymax])
        #generate
        num_graphics = 0
        chos = [0 for _ in range(n_class)]
        for _ in range(n_class):
            t = -1
            for i in range(n_class):
                if chos[i] == 0:
                    if t == -1 or boxarea(boxes[i,:]) > boxarea(boxes[t,:]):
                        t = i
            if t == -1:
                break
            chos[t] = 1
            num_graphics += 1
            for i in range(n_class):
                if chos[i] == 0:
                    if box_intersect(boxes[i,:], boxes[t,:]) / boxarea(boxes[i,:]) >= 0.2:
                        chos[i] = -1
        #NMS
        _ = 0
        for i in range(n_class):
            if chos[i] == 1:
                boxes[_, :] = boxes[i, :]
                obj_ids[_] = obj_ids[i]
                masks[_, :] = masks[i, :]
                self.draw_shape(X[i], Y[i], S[i], COLOR[i], img, masks[_, :], obj_ids[_])
                _ += 1
        maskunion = np.zeros((self.h, self.w))
        for _ in range(num_graphics - 1, -1, -1):
            masks[_, :] *= (maskunion < 0.5)
            maskunion = np.maximum(masks[_, :], maskunion)

        boxes = torch.as_tensor(boxes[:num_graphics, ], dtype=torch.float32)
        labels = torch.as_tensor(obj_ids[:num_graphics, ], dtype=torch.int64)
        masks = torch.as_tensor(masks[:num_graphics, ], dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        img = torch.tensor(img)
        img = img.permute(2,0,1)


        return img, target
        

    def __len__(self):
        return self.size

if __name__ == '__main__':
    # dataset = SingleShapeDataset(10)
    dataset = ShapeDataset(10)
    path = "../results/MaskRCNN/" 

    for i in range(10):
        imgs, labels = dataset[i]
        print(labels)
        plot_save_dataset(path+str(i)+"_data.png", imgs, labels)

