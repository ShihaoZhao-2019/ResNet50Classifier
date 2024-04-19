import os
import argparse
import torch
from traineryuan import NetworkManager
import os

def main():
    parser = argparse.ArgumentParser(
        description='Options for base model finetuning on CUB_200_2011 datasets'
    )

    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--base_lr', type=float, default=0.001,
                        help='base learning rate for training')
    parser.add_argument('--net_choice', type=str, default='ResNet',
                       help='net_choice for choosing network, whose value is in ["ResNet"]')
    parser.add_argument('--model_choice', type=int, default=50,
                        help='model_choice for choosing depth of network, whose value is in [50, 101, 152]')
    # parser.add_argument('--epochs', type=int, default=80,
    #                     help='batch size for training')
    parser.add_argument('--epochs', type=int, default=80,
                        help='batch size for training')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight_decay for SGD')
    parser.add_argument('--img_size', type=int, default=448,
                        help='image\'s size for transforms')
    parser.add_argument('--outputDirName', type=str, default='resNet50_origin',
                       help='outputDirName for  network and result')
    # parser.add_argument('--root', type=str, default='/data/kb/tanyuanyong/TransFG-master/data/CUB_200_2011',
    #                    help='root for  data')
    parser.add_argument('--root', type=str, default='/data/kb/tanyuanyong/TransFG-master/data/CUB_200_2011',
                       help='root for  data')
    # parser.add_argument('--n_easy', default=12, type=int,
    #                     help='the number of classes')
    # parser.add_argument('--n_difficult', default=4, type=int,
    #                     help='the number of samples per class')
    args = parser.parse_args()

    options = {
        'net_choice': args.net_choice,
        'model_choice': args.model_choice,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'base_lr': args.base_lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'img_size': args.img_size,
        'root':args.root,
        # 'n_easy': args.n_easy,
        # 'n_difficult': args.n_difficult,
        # 这里的设备是指多卡并行训练的时候先把数据放在的显卡索引，这里的索引是逻辑索引，当在命令行指定显卡为7,8,9,10的时候cuda:0指的就是显卡7
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        'outputDirName': args.outputDirName
    }

    manager = NetworkManager(options)
    manager.train()


if __name__ == '__main__':
    main()
