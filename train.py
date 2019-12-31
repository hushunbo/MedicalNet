'''
Training code for MRBrainS18 datasets segmentation
Written by Whalechen
'''

from setting import parse_opts 
from datasets.brains18 import BrainS18Dataset
from datasets.cbct_onthefly import CbctOnTheFlyDataset
from model import generate_model
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
from utils.logger import log
from scipy import ndimage
import SimpleITK as sitk
import os
from utils.loss import FocalLoss, seg_eval
from utils.file_process import load_lines
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils.visualize import make_image_summary


def train(train_dataloader, validate_dataloader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    # settings
    batches_per_epoch = len(train_dataloader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    loss_seg = FocalLoss()
    #loss_seg = nn.CrossEntropyLoss(ignore_index=-1)

    print("Current setting is:")
    print(sets)
    print("\n\n")     
    if not sets.no_cuda:
        loss_seg = loss_seg.cuda()
        
    train_time_sp = time.time()
    writer = SummaryWriter(sets.log_folder)

    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))
        model.train()
        scheduler.step()
        log.info('lr = {}'.format(scheduler.get_lr()))
        
        for batch_id, batch_data in enumerate(train_dataloader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            global_step = batch_id_sp + batch_id

            volumes, label_masks = batch_data
            if not sets.no_cuda: 
                volumes = volumes.cuda()
            optimizer.zero_grad()
            out_masks = model(volumes)
            # resize label
            [n, C, d, h, w] = out_masks.shape
            new_label_masks = np.zeros([n, d, h, w])
            for label_id in range(n):
                label_mask = label_masks[label_id]
                [_, ori_d, ori_h, ori_w] = label_mask.shape
                label_mask = np.reshape(label_mask, [ori_d, ori_h, ori_w])
                scale = [d*1.0/ori_d, h*1.0/ori_h, w*1.0/ori_w]
                label_mask = ndimage.interpolation.zoom(label_mask, scale, order=0)
                new_label_masks[label_id] = label_mask

            new_label_masks = torch.tensor(new_label_masks).to(torch.int64)
            if not sets.no_cuda:
                label_masks = label_masks.cuda()
                new_label_masks = new_label_masks.cuda()

            # calculating loss
            loss = loss_seg(out_masks, new_label_masks)
            loss.backward()
            optimizer.step()

            # Write current training data to tensorboard
            writer.flush()
            images_to_show = [
                volumes[0, ...],
                label_masks[0, ...],
            ]
            image_summary = make_image_summary(images_to_show, normalize=True)
            for key, value in image_summary.items():
                writer.add_image("training_volumes_" + key, value, global_step=global_step)

            out_masks = F.softmax(out_masks, dim=1)
            images_to_show = []
            for c in range(C):
                images_to_show.append(out_masks[0, [c], ...])
            image_summary = make_image_summary(images_to_show, normalize=False)
            for key, value in image_summary.items():
                writer.add_image("training_output_" + key, value, global_step=global_step)

            writer.add_scalar("training_loss", loss.item() / n, global_step=global_step)
            writer.flush()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                    'Batch: {}-{} ({}), loss_seg = {:.6f}, avg_bt= {:.3f}'\
                    .format(epoch, batch_id, batch_id_sp, loss.item(), avg_batch_time))
          
            if not sets.ci_test:
                # save model
                if (epoch+1) % save_interval == 0 and batch_id == batches_per_epoch-1:
                    model_save_path = '{}/model_epoch_{}.pth.tar'.format(save_folder, epoch)
                    model_save_dir = os.path.dirname(model_save_path)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                    
                    log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id)) 
                    torch.save({
                                'epoch': epoch,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()},
                                model_save_path)

        #             # evaluate
        #
        masks = []
        model.eval()  # for testing
        img_names = [info.split(" ")[0] for info in load_lines(sets.val_list)]
        for batch_id, batch_data in enumerate(validate_dataloader):
            # forward
            volumes, label_masks = batch_data
            if not sets.no_cuda:
                volumes = volumes.cuda()
                label_masks = label_masks.cuda()
            with torch.no_grad():
                probs = model(volumes)
                probs = F.softmax(probs, dim=1)

            # resize mask to original size
            [_, _, mask_d, mask_h, mask_w] = probs.shape
            data = sitk.ReadImage(img_names[batch_id])
            data = sitk.GetArrayFromImage(data)
            [depth, height, width] = data.shape
            mask = probs[0]
            scale = [1, depth * 1.0 / mask_d, height * 1.0 / mask_h, width * 1.0 / mask_w]
            mask = ndimage.interpolation.zoom(mask.cpu(), scale, order=1)
            mask = np.argmax(mask, axis=0)
            masks.append(mask)

        label_names = [info.split(" ")[1] for info in load_lines(sets.val_list)]
        Nimg = len(label_names)
        dices = np.zeros([Nimg, sets.n_seg_classes])
        for idx in range(Nimg):
            label = sitk.ReadImage(label_names[idx])
            label = sitk.GetArrayFromImage(label)
            print("evaluating image {}".format(idx))
            dices[idx, :] = seg_eval(masks[idx], label, range(sets.n_seg_classes))
            print(dices[idx, :])

        for idx in range(0, sets.n_seg_classes):
            mean_dice_per_task = np.mean(dices[:, idx])
            writer.add_scalar("validation_loss_label_{}".format(idx), mean_dice_per_task, global_step=epoch)
            print('mean dice for class-{} is {}'.format(idx, mean_dice_per_task))

    print('Finished training')
    if sets.ci_test:
        exit()


if __name__ == '__main__':
    # settting
    sets = parse_opts()   
    if sets.ci_test:
        sets.img_list = './toy_data/test_ci.txt' 
        sets.n_epochs = 1
        sets.no_cuda = True
        sets.data_root = './toy_data'
        sets.pretrain_path = ''
        sets.num_workers = 0
        sets.model_depth = 10
        sets.resnet_shortcut = 'A'
        sets.input_D = 14
        sets.input_H = 28
        sets.input_W = 28
       
    # getting model
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets) 
    # optimizer
    if sets.ci_test or not sets.pretrain_path:
        params = [{'params': parameters, 'lr': sets.learning_rate}]
    else:
        params = [
                { 'params': parameters['base_parameters'], 'lr': sets.learning_rate }, 
                { 'params': parameters['new_parameters'], 'lr': sets.learning_rate*100 }
                ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    # train from resume
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
              .format(sets.resume_path, checkpoint['epoch']))

    # getting data
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True    

    training_dataset = CbctOnTheFlyDataset(sets.train_list, sets)
    validate_dataset = CbctOnTheFlyDataset(sets.val_list, sets)
    train_dataloader = DataLoader(training_dataset, drop_last=True, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)
    validate_dataloader = DataLoader(validate_dataset, drop_last=False, batch_size=1, shuffle=False, num_workers=sets.num_workers, pin_memory=sets.pin_memory)

    # training
    train(train_dataloader, validate_dataloader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets)
