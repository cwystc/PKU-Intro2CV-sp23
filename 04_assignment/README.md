# Installation

The Installation is the almost the sampe as HM2. So you can **directly use the HM2 environment** or create new environment by following the below steps:

- We recommand using [Anaconda](https://www.anaconda.com/) to manage your python environments. Use the following command to create a new environment. 
```bash
conda create -n hw3 python=3.7 # use python=3.8 on Mac
conda activate hw3
```

- We recommand using [Tsinghua Mirror](https://mirrors.tuna.tsinghua.edu.cn/) to install dependent packages.

```bash
# pip
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# conda
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --add channels  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --set show_channel_urls yes
```

- Now you can install [pytorch](https://pytorch.org/get-started/previous-versions/) and other dependencies as below.
```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly # remember to remove "-c pytorh"!

# tips: always try "pip install xxx" first before "conda install xxx"
pip install opencv-python
pip install pillow
pip install tensorboardx
pip install tensorflow # for tensorboardx
pip install matplotlib # new in HM3 
```
You can also install the GPU version if you can access a GPU.

# ShapePartNet for PointNet

## Dataset
- Download and unzip ShapePartNet dataset from [here](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip).
  
```bash
wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0.zip
```
## Data Configuration
Open `HM_PointNet/utils.py`, and modify the dataset path:
```
dataset = "YOUR_PATH/shapenetcore_partanno_segmentation_benchmark_v0"
```


## Visualization

- Train network and visualize the curves
```bash
cd PointNet
python train_classification.py -d 256
cd ../exps
tensorboard --logdir .
```


# Mask RCNN
The skeleton code of Mask RCNN is based on [torchvision](https://pytorch.org/vision/stable/index.html). And you can find more detials in [https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).

You can obtain the pre-trained weight from [here](https://disk.pku.edu.cn:443/link/7ED1CE22B6BE0A3E6C8EDF3031FF4BBC).


# Submission
- Compress the the entire folder **except** dataset and checkpoints. 

- Rename the folder to **ID_PINYIN(e.g. ZhangSan)** and submit to [course.pku.edu.cn](https://course.pku.edu.cn/).

- The folder named `results` in main directoy should be structed as follows.

```bash
  results
  ├── PointNet
  └── MaskRCNN
```

- The folder named `results` in `PointNet` directoy should be structed as follows.
```bash
  PointNet
  ├── Screenshot_Classification.png 
  ├── Screenshot_Segmentation.png
  ├── 0.ply
  ├── 1.ply
  ├── 2.ply
  ├── 3.ply
  └── 4.ply
```

- The folder named `results` in `MaskRCNN` directoy should be structed as follows.
```bash
  MaskRCNN
  ├── 0_data.png
  ├── 1_data.png
  ├── 2_data.png
  ├── 3_data.png
  ├── 4_data.png
  ├── 5_data.png
  ├── 6_data.png
  ├── 7_data.png
  ├── 8_data.png
  ├── 9_data.png
  ├── 0_result.png
  ├── 1_result.png
  ├── 2_result.png
  ├── 3_result.png
  ├── 4_result.png
  ├── 5_result.png
  ├── 6_result.png
  ├── 7_result.png
  ├── 8_result.png
  ├── 9_result.png
  ├── mAP.txt
  └── maskrcnn.log
```


# Appendix and Acknowledgement
We list some libraries that may help you solve this assignment.

- [TensorboardX](https://pytorch.org/docs/stable/tensorboard.html)
- [OpenCV-Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Pillow (PIL)](https://pillow.readthedocs.io/en/stable/)
- [Torchvision.transforms](https://pytorch.org/vision/0.9/transforms.html)

Our code is inpired by the [PointNet-Pytorch](https://github.com/fxia22/pointnet.pytorch) and [detection-torchvision](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).



