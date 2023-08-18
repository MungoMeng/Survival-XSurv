# py imports
import os
import sys
import glob
import time
import cv2
import pandas as pd
import nibabel as nib
import numpy as np
import torch
from argparse import ArgumentParser
from lifelines.utils import concordance_index

# project imports
import networks
import datagenerators


def Get_survival_time(Survival_pred):

    breaks = np.array([0,300,600,900,1100,1300,1500,2100,2700,3500,6000])
    
    intervals = breaks[1:] - breaks[:-1]
    n_intervals = len(intervals)
    
    Survival_time = 0
    for i in range(n_intervals):
        cumulative_prob = np.prod(Survival_pred[0:i+1])
        Survival_time = Survival_time + cumulative_prob * intervals[i]
    
    return Survival_time


def dice(vol1, vol2, labels=None, nargout=1):
    
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)


def test(data_dir,
        test_samples,
        device,
        load_model,
        save_dir,
        save_csv_file):
    
    # prepare data files
    test_samples = np.load(test_samples, allow_pickle=True)
    
    # device handling
    if 'gpu' in device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device[-1]
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'
    
    # prepare the model
    model = networks.XSurv()
    print('loading', load_model)
    state_dict = torch.load(load_model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # survival outputs
    Patient = []
    Score = []
    
    # evaluate on testing set
    PT_inter_test = PT_union_test = 0
    MLN_inter_test = MLN_union_test = 0
    Survival_time_test = []
    Survival_label_test = []
    for test_image in test_samples:
        
        # load subject
        PET, CT, Seg_PT, Seg_MLN, Label = datagenerators.load_by_name(data_dir, test_image)
        PET = torch.from_numpy(PET).to(device).float()
        CT = torch.from_numpy(CT).to(device).float()
        Survival_label_test.append(Label)
        
        with torch.no_grad():
            pred = model(PET, CT)
        
        Seg_PT_pred = pred[0].detach().cpu().numpy().squeeze()
        _, Seg_PT_pred = cv2.threshold(Seg_PT_pred,0.5,1,cv2.THRESH_BINARY)
        PT_inter_test = PT_inter_test + np.sum(Seg_PT_pred * Seg_PT)
        PT_union_test = PT_union_test + np.sum(Seg_PT_pred + Seg_PT)
        
        Seg_MLN_pred = pred[1].detach().cpu().numpy().squeeze()
        _, Seg_MLN_pred = cv2.threshold(Seg_MLN_pred,0.5,1,cv2.THRESH_BINARY)
        MLN_inter_test = MLN_inter_test + np.sum(Seg_MLN_pred * Seg_MLN)
        MLN_union_test = MLN_union_test + np.sum(Seg_MLN_pred + Seg_MLN)
        
        Survival_pred = pred[2].detach().cpu().numpy().squeeze()
        Survival_time = Get_survival_time(Survival_pred)
        Survival_time_test.append(Survival_time)
        
        Patient.append(bytes.decode(test_image[:-4]))
        Score.append(Survival_time)
        
        # save to nii file
        if save_dir != './':
            save_path = save_dir+bytes.decode(test_image[:-4])
            print('saving to', save_path)
            
            nii_img = nib.Nifti1Image(Seg_PT_pred, np.eye(4))
            nib.save(nii_img, save_path+'_Seg_PT.nii.gz')
    
            nii_img = nib.Nifti1Image(Seg_MLN_pred, np.eye(4))
            nib.save(nii_img, save_path+'_Seg_MLN.nii.gz')
            
     
    # calculat the mean results
    Dice_PT_test = 2*PT_inter_test/PT_union_test
    Dice_MLN_test = 2*MLN_inter_test/MLN_union_test
    Dice_test = np.mean([Dice_PT_test,Dice_MLN_test])
    print('Tumor Dice: {:.3f}'.format(Dice_PT_test))
    print('Node Dice: {:.3f}'.format(Dice_MLN_test))
    print('Average Dice: {:.3f}'.format(Dice_test))
    
    Survival_label_test = np.array(Survival_label_test)
    cindex_test = concordance_index(Survival_label_test[:,0], Survival_time_test, Survival_label_test[:,1])
    print('C-index: {:.3f}'.format(cindex_test))
    
    # save to csv file
    if save_csv_file != './':
        print('saving', save_csv_file)
        df = pd.DataFrame({'Patient':Patient, 'Score':Score})
        df.to_csv(save_csv_file, index=False)
    

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        dest="data_dir", default='./',
                        help="data folder")
    parser.add_argument("--test_samples", type=str,
                        dest="test_samples", default='./',
                        help="testing samples")
    parser.add_argument("--device", type=str, default='gpu0',
                        dest="device", help="cpu or gpuN")
    parser.add_argument("--load_model", type=str,
                        dest="load_model", default='./',
                        help="load best model")
    parser.add_argument("--save_dir", type=str,
                        dest="save_dir", default='./',
                        help="save outputs to folder")
    parser.add_argument("--save_csv_file", type=str,
                        dest="save_csv_file", default='./',
                        help="save outputs to a csv file")

    args = parser.parse_args()
    test(**vars(args))
