import torch
from torchvision import transforms
import torch.nn as nn
from models.models_for_cub import ResNet
from torch.utils.data import DataLoader
import shutil
from PIL import Image
from cub import CUB
import codecs
import csv
import os

def data_write_csv(outputDirName,epoch,datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    root = './output'
    outputDir = os.path.join(root,outputDirName)
    if(os.path.isdir(outputDir) == False):
        os.makedirs(outputDir)
    file_name = os.path.join(outputDir,'epoch' + str(epoch)  + '_result.csv')
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")


class NetworkManager(object):
    def __init__(self, options):
        self.options = options
        # self.path = path
        self.device = options['device']
        print('Starting to prepare network and data...')

        # 多卡并行计算的设置，这里的设备id仅代表一开始存放数据的显卡号
        net = self._net_choice(self.options['net_choice'])
        net = nn.DataParallel(module=net)
        self.net = net.to(self.device)

        print('Network is as follows:')
        self.criterion = nn.CrossEntropyLoss()
        self.solver = torch.optim.SGD(
            self.net.parameters(), lr=self.options['base_lr'], momentum=self.options['momentum'],
            weight_decay=self.options['weight_decay']
        )
        # 动态调整学习率的库
        self.schedule = torch.optim.lr_scheduler.StepLR(self.solver, step_size=30, gamma=0.1)
        self.train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                                   transforms.RandomCrop((448, 448)),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                                  transforms.CenterCrop((448, 448)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.train_data = CUB(root=options['root'], is_train=True,
                              transform=self.train_transform)
        self.test_data = CUB(root=options['root'], is_train=False,
                             transform=self.test_transform)

        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=options['batch_size'], shuffle=False,
                                                       num_workers=2, pin_memory=True)
        print("训练集的数目为：", len(self.train_data))
        print('测试集的数目为：', len(self.test_data))

    def doTrain(self, trainDatas):
        # 训练模式
        epoch_loss = []
        num_correct = 0
        num_total = 0
        for data in trainDatas:
            imgs, labels = data
            self.solver.zero_grad()
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            output = self.net(imgs)
            loss = self.criterion(output, labels)
            epoch_loss.append(loss.data)
            pred = torch.max(output, 1)[1]
            num_correct += (pred == labels).sum()
            num_total += labels.size(0)
            loss.backward()
            # 梯度归零
            self.solver.step()
        train_acc_epoch = float(num_correct) / num_total * 100
        avg_train_loss_epoch = sum(epoch_loss) / len(epoch_loss)

        # 评估模式
        with torch.no_grad():
            self.net.eval()
            num_correct = 0
            num_total = 0
            for data in self.test_loader:
                imgs, labels = data
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                output = self.net(imgs)
                pred = torch.max(output, 1)[1]
                num_correct += (pred == labels).sum()
                num_total += labels.size(0)
            test_acc_epoch = float(num_correct) / num_total * 100
        return avg_train_loss_epoch, train_acc_epoch, test_acc_epoch, self.net

    def train(self):
        print('-' * 50)
        testaccData = [[]]
        best_train_acc = 0
        best_test_acc = 0
        self.net.train(True)
        for epoch in range(1, 8000000000000000000):
            self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.options['batch_size'], shuffle=True,
                                                            num_workers=2, pin_memory=True)
            avg_train_loss_epoch, train_acc_epoch, test_acc_epoch, model_xun = NetworkManager.doTrain(self, self.train_loader)
            testaccData.append([epoch, test_acc_epoch])
            if train_acc_epoch > best_train_acc:
                best_train_acc = train_acc_epoch
            if test_acc_epoch > best_test_acc:
                best_test_acc = test_acc_epoch
                self.save_checkpoint(self.net,1,self.options['outputDirName'])
            else:
                self.save_checkpoint(self.net,0,self.options['outputDirName'])
            print(
                '{}\t{:.4f}\t{:.2f}%\t{:.2f}%'.format(epoch, avg_train_loss_epoch, train_acc_epoch, test_acc_epoch))

        data_write_csv(self.options['outputDirName'],epoch,testaccData)


        print('best_train_acc:{:.2f}%'.format(best_train_acc))
        print('best_test_acc:{:.2f}%'.format(best_test_acc))

    def save_checkpoint(self,net, is_best, outputDirName):
        root = './output'
        outputDir = os.path.join(root, outputDirName)
        if (os.path.isdir(outputDir) == False):
            os.makedirs(outputDir)
        filename = os.path.join(outputDir,'modelSave.pkl')
        torch.save(net, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(outputDir,'model_best.pth.tar'))

    def _net_choice(self, net_choice):
        if net_choice == 'ResNet':
            return ResNet(pre_trained=True, n_class=200, model_choice=self.options['model_choice'])
        elif net_choice == 'ResNet_ED':
            return ResNet_ED(pre_trained=True, pre_trained_weight_gpu=True, n_class=200,
                             model_choice=self.options['model_choice'])
        elif net_choice == 'ResNet_SE':
            return ResNet_SE(pre_trained=True, pre_trained_weight_gpu=True, n_class=200,
                             model_choice=self.options['model_choice'])
        elif net_choice == 'ResNet_self':
            return ResNet_self(pre_trained=True, pre_trained_weight_gpu=True, n_class=200,
                               model_choice=self.options['model_choice'])

    def adjust_learning_rate(optimizer, epoch, args):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
