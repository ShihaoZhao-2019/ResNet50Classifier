import torch
import torch.nn as nn
#from models.model_resnet_ed import resnet50_ed, resnet101_ed, resnet152_ed
from utils.Config import Config
from utils.weight_init import weight_init_kaiming
#from models.model_resnet_se import se_resnet50, se_resnet101, se_resnet152
from torchvision import models
import os
import numpy as np
#from models.model_resnet import resnet50, resnet101, resnet152
"""
mask分支解耦合
"""
# Model
models.densenet121()
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    """
    resnet50 resnet101 resnet152所用的残差块的实现
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ClassResNet(nn.Module):
    """
    一个经典的残差网络的实现，每一个经典的残差网络有四个大层，只需要输入每个层间小层的数目就可以构建各种残差网络
    """
    def __init__(self, block, layers, num_chanels=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 ):
        super(ClassResNet, self).__init__()

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 每一个残差网络都是由四个大层构成，64,128,256,512这几个参数是每一层网络中前两层的卷积层数
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])
        # self.fcn = conv1x1(2048, 2000)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_chanels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn3.weight, 0)

    # 构建一个大层的残差网络，每个经典的resnet是由四个大层构成的
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 每一大层的第一次迭代都需要单独处理
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, nn.BatchNorm2d))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=nn.BatchNorm2d))

        return nn.Sequential(*layers)

def ResNet50(pretrainDir = None):
    model = ClassResNet(Bottleneck, [3, 4, 6, 3])
    if pretrainDir is not None:
        state_dict = torch.load(pretrainDir)
        print("start to load pretrain param...")
        model.load_state_dict(state_dict)
        print("load pretrain param suffessful!")
    return model

def getTextTarget(targets,textLoader):
    targetList = [i.item() for i in targets]
    textTargets = [textLoader[i] for i in targetList]
    # textTargets = torch.tensor(textTargets)
    return textTargets

class ResNet(nn.Module):
    def __init__(self, pre_trained=True, n_class=200, model_choice=50):
        super(ResNet, self).__init__()
        self.n_class = n_class
        self.base_model = self._model_choice(pre_trained, model_choice)
        self.base_model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.base_model.fc = nn.Linear(512*Config.expansion, n_class)
        self.base_model.fc.apply(weight_init_kaiming)

    def forward(self, x):
        N = x.size(0)
        assert x.size() == (N, 3, 448, 448)

        x = self.base_model(x)

        assert x.size() == (N, self.n_class)
        return x

    def _model_choice(self, pre_trained, model_choice):
        if model_choice == 50:
            # ResNet50 源码
            return models.resnet50(pre_trained)
        elif model_choice == 101:
            return models.resnet101(pretrained=pre_trained)
        elif model_choice == 152:
            return models.resnet152(pretrained=pre_trained)


