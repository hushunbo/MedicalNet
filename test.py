from setting import parse_opts 
#from datasets.brains18 import BrainS18Dataset
from datasets.cbct_onthefly import CbctOnTheFlyDataset
from model import generate_model
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy import ndimage
import SimpleITK as sitk
import nibabel as nib
import sys
import os
from utils.file_process import load_lines
import numpy as np
from utils.loss import seg_eval


# def binary_seg_eval(pred, label):
#     # only one class
#     dices = np.zeros(1)
#     s = pred + label
#     inter = len(np.where(s >= 2)[0])
#     conv = len(np.where(s >= 1)[0]) + inter
#     try:
#         dices[0] = 2.0 * inter / conv
#     except:
#         print("conv is zeros when dice = 2.0 * inter / conv")
#         dices[0] = 0
#     print(dices)
#     return dices


def test(data_loader, model, img_names, sets):
    masks = []
    model.eval() # for testing 
    for batch_id, batch_data in enumerate(data_loader):
        # forward
        volume = batch_data
        if not sets.no_cuda:
            volume = volume.cuda()
        with torch.no_grad():
            probs = model(volume)
            probs = F.softmax(probs, dim=1)

        # resize mask to original size
        [batchsize, _, mask_d, mask_h, mask_w] = probs.shape
        # data = nib.load(img_names[batch_id])
        data = sitk.ReadImage(img_names[batch_id])
        print("processing image {}".format(img_names[batch_id]))
        # data = data.get_data()
        data = sitk.GetArrayFromImage(data)
        [depth, height, width] = data.shape
        mask = probs[0]
        scale = [1, depth*1.0/mask_d, height*1.0/mask_h, width*1.0/mask_w]
        mask = ndimage.interpolation.zoom(mask.cpu().numpy(), scale, order=1)
        mask = np.argmax(mask, axis=0)

        masks.append(mask)
 
    return masks


if __name__ == '__main__':
    # settting
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'

    # getting model
    checkpoint = torch.load(sets.resume_path)
    net, _ = generate_model(sets)
    net.load_state_dict(checkpoint['state_dict'])

    # data tensor
    testing_data = CbctOnTheFlyDataset(sets.val_list, sets)
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    # testing
    img_names = [info.split(" ")[0] for info in load_lines(sets.val_list)]
    masks = test(data_loader, net, img_names, sets)
    
    # calculate dice
    label_names = [info.split(" ")[1] for info in load_lines(sets.val_list)]
    Nimg = len(label_names)
    dices = np.zeros([Nimg, sets.n_seg_classes])
    for idx in range(Nimg):
        #label = nib.load(label_names[idx])
        #label = label.get_data()
        label = sitk.ReadImage(label_names[idx])
        label = sitk.GetArrayFromImage(label)
        print("evaluating image {}".format(idx))
        dices[idx, :] = seg_eval(masks[idx], label, range(sets.n_seg_classes))
        #dices[idx, :] = binary_seg_eval(masks[idx], label)
        print(dices[idx, :])
    
    # print result
    for idx in range(0, sets.n_seg_classes):
        mean_dice_per_task = np.mean(dices[:, idx])
        print('mean dice for class-{} is {}'.format(idx, mean_dice_per_task))   
