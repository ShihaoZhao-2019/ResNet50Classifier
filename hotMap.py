import os
import json
from os.path import join
import sys
import numpy as np
import scipy
from scipy import io
import imageio
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
# import cv2

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import heapq
import torchvision.models as models
import cv2
import matplotlib.pyplot as plt

"""
为了可视化更改一下dataloader
"""



os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class CUB():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))

        # 这里选择训练和测试文件
        #  istrain 选择训练文件 否则选择测试文件
        if self.is_train:
            train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        else:
            train_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        # print(train_file_list)
        # print(test_file_list)
        self.train_img = [imageio.imread(os.path.join(self.root, 'images', train_file)) for train_file in
                          train_file_list[:data_len]]
        self.train_img_path = [os.path.join(self.root, 'images', train_file) for train_file in
                          train_file_list[:data_len]]
        self.train_img_name = [train_file for train_file in
                          train_file_list[:data_len]]
        if self.is_train:
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        else:
            self.train_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
        self.train_img_name = [x for x in train_file_list[:data_len]]

    def __getitem__(self, index):
        img, target = self.train_img[index], self.train_label[index]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img,self.train_img_path[index],self.train_img_name[index], target

    def __len__(self):
            return len(self.train_label)



class getFeatureMap(nn.Module):
    def __init__(self,modelDir):
        super(getFeatureMap, self).__init__()
        net = torch.load(modelDir)  # 载入整个模型
        # 可能是并行计算的原因，所以这里要通过module来访问ResNetText子网络
        self.base_model = net.module.base_model
        pass

    def forward(self,x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        y = self.base_model.avgpool(x)
        feature = x
        y = y.view(y.size(0), -1)
        y = self.base_model.fc(y)
        return y,feature

def getDataLoader(root,train):

    imageTransform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                              transforms.CenterCrop((448, 448)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    myData = CUB(root=root, is_train=train,transform=imageTransform)

    myLoader = torch.utils.data.DataLoader(myData, batch_size=1, shuffle=False,
                                                           num_workers=0, pin_memory=True)
    return myLoader

def drawFeature(isTrain,features,imgOrignPath,imgName,labels,modelName,drawClassFeature,label,pred):
    # heat 为某层的特征图，自己手动获取
    heat = features.data.cpu().numpy()  # 将tensor格式的feature map转为numpy格式
    heat = np.squeeze(heat, 0)  # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除
    if(drawClassFeature  == True):
        label = labels.data.cpu().numpy()[0]
        heat = heat[label * 10:label * 10 + 10, :] # 切片获取某几个通道的特征图
    heatmaps = np.maximum(heat, 0)  # heatmap与0比较
    # channelNumber = features.shape[1]
    if(isTrain):
        root = './output/seeAttention/' + modelName + '/train'
    else:
        root = './output/seeAttention/' + modelName + '/test'
    # saveFolder = os.path.join(root, imgName[0][0:-4])
    saveFolder = root
    if os.path.isdir(saveFolder) is False:
        os.makedirs(saveFolder)

    img = cv2.imread(imgOrignPath[0])
    heatmapSum = np.sum(heatmaps, axis=0)  # 多通道时，取均值
    heatmapSum/=np.max(heatmapSum)
    heatmapSum = cv2.resize(heatmapSum, (img.shape[1], img.shape[0]))
    heatmapSum = np.uint8(255 * heatmapSum)
    heatmapSum = cv2.applyColorMap(heatmapSum, cv2.COLORMAP_JET)
    heat_img = cv2.addWeighted(img, 1, heatmapSum, 0.5, 0)
    savePath = os.path.join(saveFolder, "label_" + str(label)+ "pred_" + str(pred) +"_" +imgName[0][5:])
    cv2.imwrite(savePath,heat_img)


def main(modelName,root,train,modelDir):
    net = getFeatureMap(modelDir)
    net.cuda()
    net.eval()
    net.cuda()
    Imageloader = getDataLoader(root,train)


    with torch.no_grad():
        net.eval()
        num_correct = 0
        num_total = 0
        for imgs,imgOrignPath,imgName,labels in Imageloader:
            imgs = imgs.cuda()
            labels = labels.cuda().item()
            output,feature = net(imgs)
            pred = torch.max(output, 1)[1].item()
            if(labels != pred):
                drawFeature(train, feature, imgOrignPath, imgName, labels, modelName, False,labels,pred)
            else:
                drawFeature(train, feature, imgOrignPath, imgName, labels, modelName, False,labels,pred)
                num_correct+=1
            num_total+=1
            print(num_total,"  test_acc_epoch:",str(num_correct/num_total * 100),"%")

        #     pred = torch.max(output, 1)[1]
        #     print(pred)
        #     num_correct += (pred == labels).sum()
        #     num_total += labels.size(0)
        # test_acc_epoch = float(num_correct) / num_total * 100
        # print(test_acc_epoch)


"""
模型最大准确率83.24
"""
if __name__ == '__main__':
    root = '/data/kb/zhaoshihao/TEA_CUB_SMALL'
    modelName = 'resNet50TeaLavel'
    train = False
    modelDir = '/data/kb/zhaoshihao/ResNet50Classifier/output/resNet50_origin/model_best.pth.tar'
    main(modelName,root,train,modelDir)


