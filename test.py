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
    probs = []
    model.eval() # for testing 
    for batch_id, batch_data in enumerate(data_loader):
        # forward
        volume = batch_data
        if not sets.no_cuda:
            volume = volume.cuda()
        with torch.no_grad():
            prob = model(volume)
            prob = F.softmax(prob, dim=1)

        # resize mask to original size
        [batchsize, _, mask_d, mask_h, mask_w] = prob.shape
        data = sitk.ReadImage(img_names[batch_id])
        print("processing image {}".format(img_names[batch_id]))
        data = sitk.GetArrayFromImage(data)
        [depth, height, width] = data.shape
        prob = prob[0]
        scale = [1, depth*1.0/mask_d, height*1.0/mask_h, width*1.0/mask_w]
        prob = ndimage.interpolation.zoom(prob.cpu().numpy(), scale, order=1)
        mask = np.argmax(prob, axis=0)

        masks.append(mask)
        probs.append(prob)
 
    return masks, probs


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
    masks, probs = test(data_loader, net, img_names, sets)
    
    # calculate dice
    label_names = [info.split(" ")[1] for info in load_lines(sets.val_list)]
    Nimg = len(label_names)
    dices = np.zeros([Nimg, sets.n_seg_classes])
    for idx in range(Nimg):
        label_itk = sitk.ReadImage(label_names[idx])
        label_arr = sitk.GetArrayFromImage(label_itk)

        image_file = os.path.join(sets.save_folder, 'image_{}.nii.gz'.format(idx))
        label_file = os.path.join(sets.save_folder, 'label_{}.nii.gz'.format(idx))
        os.system('cp {} {}'.format(img_names[idx], image_file))
        os.system('cp {} {}'.format(label_names[idx], label_file))
        print("evaluating image {}".format(idx))
        mask_arr = masks[idx]
        prob_arr = probs[idx]
        dices[idx, :] = seg_eval(mask_arr, label_arr, range(sets.n_seg_classes))
        print(dices[idx, :])
        mask_itk = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
        mask_itk.CopyInformation(label_itk)
        mask_file = os.path.join(sets.save_folder, 'mask_{}.nii.gz'.format(idx))
        sitk.WriteImage(mask_itk, mask_file)

        C = prob_arr.shape[0]
        for c in range(C):
            prob_c_arr = prob_arr[c, ...]
            prob_c_itk = sitk.GetImageFromArray(prob_c_arr)
            prob_c_itk.CopyInformation(label_itk)
            prob_c_file = os.path.join(sets.save_folder, 'prob_{}_{}.nii.gz'.format(c, idx))
            sitk.WriteImage(prob_c_itk, prob_c_file)

    # print result
    for idx in range(0, sets.n_seg_classes):
        mean_dice_per_task = np.mean(dices[:, idx])
        print('mean dice for class-{} is {}'.format(idx, mean_dice_per_task))   
