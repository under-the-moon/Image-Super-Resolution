import argparse
import os

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--save_path', type=str, default='./weights', help='saving path that is a folder')
parser.add_argument('--sample_path', type=str, default='./samples', help='training samples path that is a folder')
parser.add_argument('--log_dir', type=str, default='./run/logs', help='training samples path that is a folder')

# Training parameters
parser.add_argument('--epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--resume', action='store_true', default=True)
parser.add_argument('--resume_epoch', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=1e-3, help='Adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='Adam: beta 1')
parser.add_argument('--b2', type=float, default=0.999, help='Adam: beta 2')
parser.add_argument('--weight_decay', type=float, default=5e-6, help='Adam: weight decay')
parser.add_argument('--lr_decrease_epoch', type=int, default=5, help='lr decrease at certain epoch and its multiple')
parser.add_argument('--lr_decrease_factor', type=float, default=0.5,
                    help='lr decrease factor, for classification default 0.1')
# parser.add_argument('--num_workers', type=int, default=8, help='number of cpu threads to use during batch generation')
# num_workers 为其他正整数 在windows下容易发生死锁等待
parser.add_argument('--num_workers', type=int, default=0, help='number of cpu threads to use during batch generation')

# Network parameters
parser.add_argument('--in_channels', type=int, default=3, help='input RGB image + 1 channel mask')
parser.add_argument('--embed_dim', type=int, default=96, help='model embed_dim')
parser.add_argument('--scale', type=int, default=2, help='model scale [2, 4]')
parser.add_argument('--num_patchs', type=int, default=64, help='num patches')
parser.add_argument('--sf_layer', type=int, default=2, help='shallow feature')

# Dataset parameters
parser.add_argument('--baseroot', type=str, default="E:\\data\\ir\\baidu", help='the training folder')
# parser.add_argument('--baseroot', type=str, default="E:\\data\\ir\\DIV2K_train_LR_bicubic", help='the training folder')
parser.add_argument('--img_size', type=int, default=32, help='silr_decrease_epochze of image')

if __name__ == '__main__':
    opt = parser.parse_args()
    print('Net params: ')
    print(opt)

    import train_helper as helper

    helper.train(opt)
