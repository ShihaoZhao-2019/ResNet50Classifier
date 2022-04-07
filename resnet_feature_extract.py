# encoding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import heapq
import torchvision.models as models


def extractor(img_path, net, use_gpu):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )

    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    # print(img.shape)
    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    # print(x.shape)
    if use_gpu:
        x = x.cuda()
        net = net.cuda()

    y = net.module.base_model.conv1(x)
    y = net.module.base_model.bn1(y)
    y = net.module.base_model.relu(y)
    y = net.module.base_model.maxpool(y)
    y = net.module.base_model.layer1(y)
    y = net.module.base_model.layer2(y)
    y = net.module.base_model.layer3(y)
    y = net.module.base_model.layer4(y)

    y = net.module.base_model.avgpool(y)
    y = y.cpu()
    y = torch.squeeze(y)
    y = y.data.numpy()
    # print(y)
    # print(y.shape)
    y = y.reshape(2048, 1)
    return y


def Resnet_Nei_similarity(lei_matrix):
    column = lei_matrix.shape[1]
    count = 0
    for ii in range(1, column):
        for jj in range(ii + 1, column):
            NEI_feature1 = lei_matrix[:, [ii]]
            NEI_feature2 = lei_matrix[:, [jj]]
            liang_ou = np.sqrt(np.sum(np.square(NEI_feature1 - NEI_feature2)))
            count = count + liang_ou
    number = column * (column - 1) / 2
    NEI_similarity = count / number
    hanghe = np.sum(lei_matrix, axis=1)
    # print("每一行相加后结果为{}", format(hanghe))
    for j in range(0, 2048):
        hanghe[j] = hanghe[j] / column
    return NEI_similarity, hanghe


def Resnet_Between_similarity(lei_average):
    row = len(lei_average)
    print(row)
    between_similarity = np.zeros((row, row))
    for r in range(0, row):
        for s in range(r + 1, row):
            average_feature1 = lei_average[r]
            average_feature2 = lei_average[s]
            liang_ou = np.sqrt(np.sum(np.square(average_feature1 - average_feature2)))
            between_similarity[[r], [s]] = liang_ou
            between_similarity[[s], [r]] = liang_ou
    return between_similarity


def solve_diff(lei_zhi, jian_zhi, diff_number):
    CR = []
    # txt文档中记录着每个类别中与之相似的类别，每读一行，则得到这一行的类内相似度和类间相似度
    file = open('./CUB_similar_lei.txt')
    i = 0
    for line in file.readlines():
        curLine = line.strip().split(" ")
        b = len(curLine)
        average_jian = 0
        if curLine[0] == 0:
            return average_jian
        else:
            for j in range(0, b):
                jin_lei = int(curLine[j])
                average_jian = average_jian + float(jian_zhi[[i], [jin_lei - 1]])
            average_jian = average_jian / b
        i_lei = lei_zhi[i]
        xiangsibi = average_jian / i_lei
        CR.append(xiangsibi)
        # print("所有类别的CR值为：{}", format(CR))
        i = i + 1
    diff_lei = list(map(CR.index, heapq.nlargest(diff_number, CR)))
    # max_CR = heapq.nlargest(diff_number, CR)
    zong = [i for i in range(200)]
    easy_lei = list(set(zong) - set(diff_lei))
    return diff_lei, easy_lei


def main(model, diff_lei_number):
    model = model.cuda()
    print("调用Net的是：", format(model))
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    # 以下是得到鸟类数据集中所有的按顺序排序的200个类别的路径
    data_dir = './CUB_200_2011/images'
    files_list1 = [x for x in range(0, 200)]
    file_parent = os.listdir(data_dir)
    for file in file_parent:
        lei_number = int(file[0:3]) - 1
        file_path = os.path.join(data_dir, file)
        files_list1[lei_number] = file_path
    print("所有类别的路径{}", format(files_list1))
    NEI_similarity = []
    JIAN = []
    # 得到某个类别中所有图片的路径
    for x_path in files_list1:
        files_list2 = []
        file_picture = os.listdir(x_path)
        for file in file_picture:
            file_path = os.path.join(x_path, file)
            files_list2.append(file_path)
        # print(files_list2)
        use_gpu = torch.cuda.is_available()
        # 初始化所有图片的特征矩阵为0，依次得到某个类别中所有图像的特征向量，然后添加到矩阵当中去
        picture_number = len(files_list2)
        lei_feature = np.zeros((2048, picture_number))
        i = 0
        for y_path in files_list2:
            # print("x_path" + x_path)
            # file_name = y_path.split('/')[-1]
            # print(fx_path)
            temp = extractor(y_path, model, use_gpu)
            lei_feature[:, [i]] = temp
            i = i + 1
        # print("一个类别所有图片的特征：{}", format(lei_feature))
        nei, jian = Resnet_Nei_similarity(lei_feature)
        NEI_similarity.append(nei)
        JIAN.append(jian)
    # print("所有类别的类内相似度为{}", format(NEI_similarity))
    # print("所有类别的平均特征向量为{}", format(JIAN))
    Between_similarity = Resnet_Between_similarity(JIAN)
    # print("所有类别的类间相似度为{}", format(Between_similarity))
    Difficult_lei, Easy_lei = solve_diff(NEI_similarity, Between_similarity, diff_lei_number)
    print("最困难类别的索引值为：{}", format(Difficult_lei))
    print("最容易类别的索引值为：{}", format(Easy_lei))
    return Difficult_lei, Easy_lei

