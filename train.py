# author:oldpan
# data:2018-4-16
# Just for study and research

from __future__ import print_function
import argparse
import os

import cv2
import numpy as np
import torch

import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

from models import Autoencoder, toTensor, var_to_np
from util import get_image_paths, load_images, stack_images
from training_data import get_training_data

parser = argparse.ArgumentParser(description='DeepFake-Pytorch')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda is True:
    print('===> Using GPU to train')
else:
    print('===> Using CPU to train')

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print('===> Loaing datasets')
images_A = get_image_paths("data/trump")
images_B = get_image_paths("data/cage")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0
images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))

model = Autoencoder()

print('===> Try resume from checkpoint')
if os.path.isdir('checkpoint'):
    try:
        checkpoint = torch.load('./checkpoint/autoencoder.t7')
        model.load_state_dict(checkpoint['state'])
        start_epoch = checkpoint['epoch']
        print('===> Load last checkpoint data')
    except FileNotFoundError:
        print('Can\'t found autoencoder.t7')
else:
    start_epoch = 0
    print('===> Start from scratch')

if args.cuda:
    device = torch.device('cuda:0')
    cudnn.benchmark = True

criterion = nn.L1Loss()
optimizer_1 = optim.Adam([{'params': model.encoder.parameters()},
                          {'params': model.decoder_A.parameters()}]
                         , lr=5e-5, betas=(0.5, 0.999))
optimizer_2 = optim.Adam([{'params': model.encoder.parameters()},
                          {'params': model.decoder_B.parameters()}]
                         , lr=5e-5, betas=(0.5, 0.999))

# print all the parameters im model
# s = sum([np.prod(list(p.size())) for p in model.parameters()])
# print('Number of params: %d' % s)

if __name__ == "__main__":

    print('Start training, press \'q\' to stop')

    for epoch in range(start_epoch, args.epochs):
        batch_size = args.batch_size

        warped_A, target_A = get_training_data(images_A, batch_size)
        warped_B, target_B = get_training_data(images_B, batch_size)

        warped_A, target_A = toTensor(warped_A), toTensor(target_A)
        warped_B, target_B = toTensor(warped_B), toTensor(target_B)

        if args.cuda:
            warped_A = warped_A.to(device).float()
            target_A = target_A.to(device).float()
            warped_B = warped_B.to(device).float()
            target_B = target_B.to(device).float()

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        model.to(device)
        warped_A = model(warped_A, 'A')
        warped_B = model(warped_B, 'B')

        loss1 = criterion(warped_A, target_A)
        loss2 = criterion(warped_B, target_B)
        loss = loss1.item() + loss2.item()
        loss1.backward()
        loss2.backward()
        optimizer_1.step()
        optimizer_2.step()
        print('epoch: {}, lossA:{}, lossB:{}'.format(epoch, loss1.item(), loss2.item()))

        if epoch % args.log_interval == 0:

            test_A_ = target_A[0:14]
            test_B_ = target_B[0:14]
            test_A = var_to_np(target_A[0:14])
            test_B = var_to_np(target_B[0:14])
            print('===> Saving models...')
            state = {
                'state': model.state_dict(),
                'epoch': epoch
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/autoencoder.t7')

        figure_A = np.stack([
            test_A,
            var_to_np(model(test_A_, 'A')),
            var_to_np(model(test_A_, 'B')),
        ], axis=1)
        figure_B = np.stack([
            test_B,
            var_to_np(model(test_B_, 'B')),
            var_to_np(model(test_B_, 'A')),
        ], axis=1)

        figure = np.concatenate([figure_A, figure_B], axis=0)
        figure = figure.transpose((0, 1, 3, 4, 2))
        figure = figure.reshape((4, 7) + figure.shape[1:])
        figure = stack_images(figure)

        figure = np.clip(figure * 255, 0, 255).astype('uint8')

        cv2.imshow("", figure)
        key = cv2.waitKey(1)
        if key == ord('q'):
            exit()
