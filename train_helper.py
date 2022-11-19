import os
import time
import datetime
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.bddataset import ISRDataset, collate_fn
# from dataset.isrdataset import ISRDataset, collate_fn
from net.swin_ca import MixSwinCa
from tools.img_util import save_sample_png
from tools.metric import pnsr
from loss import ISRLoss

from tensorboardX import SummaryWriter


def train(opt):
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    log_dir = opt.log_dir
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # l1_func = nn.L1Loss()
    # sl1_func = nn.SmoothL1Loss()
    loss_func = ISRLoss()

    model = MixSwinCa(img_size=opt.img_size, in_channels=opt.in_channels, embed_dim=opt.embed_dim,
                      sf_layer=opt.sf_layer)

    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

    def adjust_learning_rate(lr_in, optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = max(lr_in * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch)), 1e-6)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('The lr successfully change from %s to %s at epoch %s' % (lr_in, lr, epoch))
        return lr

    # Save model
    def save_model(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'isr_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        torch.save(net.state_dict(), model_name)
        print('The trained model is successfully saved at epoch %d' % (epoch))

    # load the model

    def load_model(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'isr_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        pretrained_dict = torch.load(model_name)
        net.load_state_dict(pretrained_dict, strict=False)

    if opt.resume:
        load_model(model, opt.resume_epoch, opt)
        print('--------------------Pretrained Models are Loaded--------------------')

    dataset = ISRDataset(data_path=opt.baseroot, scale=opt.scale, img_size=opt.img_size, num_patchs=opt.num_patchs)
    print('The train number of images equals to %d' % len(dataset))

    train_loader = DataLoader(dataset, batch_size=opt.batch_size, collate_fn=collate_fn, shuffle=True,
                              num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    writer = SummaryWriter(log_dir)

    gloab_step = 0

    # Training loop
    for epoch in range(opt.resume_epoch, opt.epochs):

        epoch_losses = []
        nspr_valuess = []

        lr = adjust_learning_rate(opt.lr, optimizer, (epoch + 1), opt)
        model.train()

        val_dataset = []

        for batch_idx, (img, target) in enumerate(train_loader):
            img = img.float().cuda()
            target = target.float().cuda()

            optimizer.zero_grad()

            out = model(img)

            loss, coarse_loss, points_loss = loss_func(out, target)

            loss.backward()

            optimizer.step()

            psnr_value = pnsr(out['pred'], target)

            epoch_losses.append(loss)
            nspr_valuess.append(psnr_value)

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [ISR Loss: %.5f] [Coarse Loss: %.5f] [Points Loss: %.5f] [PSNR: %.5f] [LR: %.6f]" %
                  (epoch + 1,
                   opt.epochs,
                   batch_idx + 1,
                   len(train_loader),
                   loss.item(),
                   coarse_loss.item(),
                   points_loss.item(),
                   psnr_value,
                   lr
                   ))

            if (batch_idx + 1) % 40 == 0:
                pred = out['pred']
                with torch.no_grad():
                    out = model(img, training=False)
                fine = out['fine']
                img_list = [target, pred, fine]
                name_list = ['target', 'pred', 'fine']
                save_sample_png(sample_folder=sample_folder, sample_name='epoch%d_batch%d' % (epoch + 1, batch_idx + 1),
                                img_list=img_list, name_list=name_list, pixel_max_cnt=255)
                val_dataset.append((batch_idx, img))

            gloab_step += 1
            writer.add_scalar('loss', loss, gloab_step)
            writer.add_scalar('psnr', psnr_value, gloab_step)

        # epoch_loss = np.mean(epoch_losses)
        # nspr_value = np.mean(nspr_valuess)
        #
        # writer.add_scalar('epoch_loss', epoch_loss, epoch)
        # writer.add_scalar('nspr_value', nspr_value, epoch)

        # Save the model
        save_model(model, (epoch + 1), opt)