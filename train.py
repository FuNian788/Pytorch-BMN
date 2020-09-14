# coding: utf-8

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import MyDataset
from loss import bmn_loss, get_mask
from model import BMN_model
from opt import MyConfig
from utils.opt_utils import get_cur_time_stamp

# GPU setting.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # range GPU in order
os.environ["CUDA_VISIBLE_DEVICES"] = "3"            

# Basic test.
print("Pytorch's version is {}.".format(torch.__version__))
print("CUDNN's version is {}.".format(torch.backends.cudnn.version()))
print("CUDA's state is {}.".format(torch.cuda.is_available()))
print("CUDA's version is {}.".format(torch.version.cuda))
print("GPU's type is {}.".format(torch.cuda.get_device_name(0)))

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

# Weights initialization.
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)

# Random seed.
def my_worker_init_fn():
    seed = torch.initial_seed()
    np_seed = seed // (2 ** 32 - 1)
    random.seed(seed)                 
    np.random.seed(np_seed)     
    # torch.manual_seed(seed)             # CPU.
    # torch.cuda.manual_seed(seed)        # One GPU.
    # torch.cuda.manual_seed_all(seed)    # Multiple GPUs.
    # torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    opt = MyConfig()
    opt.parse()

    start_time = str(get_cur_time_stamp())

    if not os.path.exists(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    model = BMN_model(opt)
    model = nn.DataParallel(model).cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.step_gamma)

    if opt.train_from_checkpoint: 
        checkpoint = torch.load(opt.checkpoint_path + '9_param.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 1

    train_dataset = MyDataset(opt)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, 
                                                   num_workers=opt.num_workers, pin_memory=True)

    valid_dataset = MyDataset(opt)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True, 
                                                   num_workers=opt.num_workers, pin_memory=True)    

    valid_best_loss = float('inf')

    for epoch in tqdm(range(start_epoch, opt.epochs + 1)):

        # Train.
        model.train()
        torch.cuda.empty_cache()
        epoch_train_loss = 0

        for train_iter, train_data in tqdm(enumerate(train_dataloader, start=1)):
            
            optimizer.zero_grad()
            
            video_feature, gt_iou_map, start_score, end_score = train_data
            video_feature = video_feature.cuda()
            gt_iou_map = gt_iou_map.cuda()
            start_score = start_score.cuda()
            end_score = end_score.cuda()

            bm_confidence_map, start, end = model(video_feature)

            bm_mask = get_mask(opt.temporal_scale).cuda()
            # train_loss: total_loss, tem_loss, pem_reg_loss, pem_cls_loss
            train_loss = bmn_loss(bm_confidence_map, start, end, gt_iou_map, start_score, end_score, bm_mask)
            train_loss[0].backward()

            optimizer.step()

            epoch_train_loss = epoch_train_loss + train_loss[0].item()

        scheduler.step()

        # Valid.
        epoch_valid_loss = 0
        with torch.no_grad():
            model.eval()
            for valid_iter, valid_data in enumerate(valid_dataloader, start=1):
                video_feature, gt_iou_map, start_score, end_score = valid_data
                video_feature = video_feature.cuda()
                gt_iou_map = gt_iou_map.cuda()
                start_score = start_score.cuda()
                end_score = end_score.cuda()

                bm_confidence_map, start, end = model(video_feature)

                valid_loss = bmn_loss(bm_confidence_map, start, end, gt_iou_map, start_score, end_score, bm_mask)

                epoch_valid_loss = epoch_valid_loss + valid_loss[0].item()

        if epoch <= 10 or epoch % 5 == 0:
            print('Epoch {}: Training loss {:.3}, Validation loss {:.3}'.format(
                    epoch, float(epoch_train_loss/train_iter), float(epoch_valid_loss/valid_iter)))  
            with open(opt.save_path + start_time + '/log.txt', 'a') as f:
                f.write('Epoch {}: Training loss {:.3}, Validation loss {:.3} \n'.format(
                    epoch, float(epoch_train_loss/train_iter), float(epoch_valid_loss/valid_iter)))  


        if epoch_valid_loss < valid_best_loss:
            # Save parameters.
            checkpoint = {'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'epoch': epoch}
            torch.save(checkpoint, opt.save_path + start_time + '/' + str(epoch) + '_param.pth.tar')
            valid_best_loss = epoch_valid_loss
            
            # Save whole model.
            # torch.save(model, opt.save_path)
