from unittest.mock import patch
import numpy as np
import torch
from medpy import metric
# from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
import os
from tqdm import tqdm
from PIL import Image

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1)) # -> [B, 1, H, W]
        output_tensor = torch.cat(tensor_list, dim=1) # -> [B, 2, H, W]
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

'''Calculate F1, Recall, Precision, accuracy'''
def get_ConfMax(prediction, groundtruth, pos, neg):
    TP = np.sum(np.logical_and(np.equal(prediction, pos), np.equal(groundtruth, pos)))
    FP = np.sum(np.logical_and(np.equal(prediction, pos), np.equal(groundtruth, neg)))
    TN = np.sum(np.logical_and(np.equal(prediction, neg), np.equal(groundtruth, neg)))
    FN = np.sum(np.logical_and(np.equal(prediction, neg), np.equal(groundtruth, pos)))
    precision = TP / (TP + FP) if (TP+FP) != 0 else None
    recall = TP / (TP + FN) if (TP+FN) != 0 else None
    F1_score = 2 * TP / (2 * TP + FP + FN) if (2*TP+FP+FN) != 0 else None
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP+FP+TN+FN) != 0 else None

    po = ( TP + TN ) / ( TP + FN + FP + TN )
    pe = (( TP + FP )*( TP + FN ) + ( FN + TN )*( FP + TN )) / ( TP+FN+TN+FP )**2
    Kappa = (po-pe) / (1-pe) if (1-pe) != 0 else None
    return precision, recall, F1_score, accuracy, Kappa


'''Calculate IoU (also called Jaccard Index)'''
def get_IoU(prediction, groundtruth, pos):
    overlap = np.logical_and(np.equal(prediction, pos), np.equal(groundtruth, pos))
    union = np.logical_or(np.equal(prediction, pos), np.equal(groundtruth, pos))
    IoU = np.sum(overlap) / np.sum(union) if np.sum(union) != 0 else None
    return IoU

'''Calculate HD (Hausdroff Distance)'''
def get_HD(pre, tar, reversal=False):
    # reverse or not
    if reversal == True:
        pre_f = 255 - pre
        tar_f = 255 - tar
    else:
        pre_f = pre
        tar_f = tar

    # if image contains nothing, return None and break
    if np.count_nonzero(pre_f) == 0 or np.count_nonzero(tar_f) == 0:
        return None

    hd = metric.binary.hd(pre_f, tar_f)

    return hd

'''ASD'''
def get_ASD(pre, tar, reversal=False):
    # reverse or not
    if reversal == True:
        pre_f = 255 - pre
        tar_f = 255 - tar
    else:
        pre_f = pre
        tar_f = tar
    
    # get img
    if np.count_nonzero(pre_f) == 0 or np.count_nonzero(tar_f) == 0:
        return None
    
    asd = metric.binary.asd(pre_f, tar_f)

    return asd

'''ASSD'''
def get_ASSD(pre, tar, reversal=False):
    # reversal or not
    if reversal == True:
        pre_f = 255 - pre
        tar_f = 255 - tar
    else:
        pre_f = pre
        tar_f = tar
    
    # get img
    if np.count_nonzero(pre_f) == 0 or np.count_nonzero(tar_f) == 0:
        return None
    
    assd = metric.binary.assd(pre_f, tar_f)

    return assd


def save_case(
        image: torch.Tensor, 
        label: torch.Tensor, 
        net, 
        classes: int, 
        test_save_path: str=None, 
        case: str=None
    ):
    label = label.squeeze(0).cpu().detach().numpy()
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(image.cuda())[-1], dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()

    if test_save_path is not None:
        prediction[prediction==1] = 255
        cv2.imwrite((test_save_path+'/'+ case), prediction)

def ave(list):
    """
    Use for calculating average
    """
    return sum(list)/len(list)

def convert_to_binary(label):
    label[label != 29] = 0 # 29 is the building class pixel value
    label[label == 29] = 255
    return label

def evaluation(data_root, pred_dir):
    precs = list()
    recs = list()
    accs = list()
    f1_scores = list()
    IoUs = list()
    hds = list()
    Kappas = list()
    tar_dir = os.path.join(data_root, 'labels')
    for root, dirs, files in os.walk(pred_dir):
        for file in tqdm(files):
            assert file.endswith('.tif'), "The file is not a tif file!"
            pre = np.array(Image.open(os.path.join(pred_dir, file)).convert('L'))
            tar = convert_to_binary(np.array(Image.open(os.path.join(tar_dir, file)).convert('L')))
            prec, rec, f1, acc, Kappa = get_ConfMax(pre, tar, pos=255, neg=0)
            IoU = get_IoU(pre, tar, pos=255)
            hd = get_HD(pre, tar, reversal=True)
            if not hd or not f1 or not IoU or not Kappa:
                continue
            else:
                precs.append(prec)
                recs.append(rec)
                accs.append(acc)
                f1_scores.append(f1)
                IoUs.append(IoU)
                hds.append(hd)
                Kappas.append(Kappa)
            # print("{} is over!".format(file))
    result = 'Average: F1:{} IoU:{} HD:{} Kappa:{} Prec:{} Rec:{} Acc:{}'.format(
        ave(f1_scores), ave(IoUs), ave(hds), ave(Kappas), ave(precs), ave(recs), ave(accs)
        )

    print(result)