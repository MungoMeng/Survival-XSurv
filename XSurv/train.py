# py imports
import os
import glob
import sys
import random
import time
import torch
import cv2
import numpy as np
from argparse import ArgumentParser
from lifelines.utils import concordance_index

# project imports
import datagenerators
import networks
import losses


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


def lr_scheduler(epoch):

    if epoch < 20:
        lr = 1e-4
    elif epoch < 40:
        lr = 5e-5
    elif epoch < 60:
        lr = 1e-5
    else:
        lr = 1e-6
    return lr


def train(data_dir,
          train_samples,
          valid_samples,
          model_dir,
          load_model,
          device,
          initial_epoch,
          epochs,
          steps_per_epoch,
          batch_size):
     
    # prepare data files
    train_samples = np.load(train_samples, allow_pickle=True)
    valid_samples = np.load(valid_samples, allow_pickle=True)
    
    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    
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
    if load_model != './':
        print('loading', load_model)
        state_dict = torch.load(load_model, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device) 
    
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters())
    
    # prepare losses
    Losses = [losses.Seg_loss, losses.Seg_loss, losses.Loglike_loss, losses.L2_Reg_loss]
    Weights = [1.0, 1.0, 1.0, 1.0]
    
    # data generator
    data_gen = datagenerators.gen_load(data_dir, train_samples, batch_size=batch_size, balance_censored_sample=True)
    train_gen = datagenerators.gen_process(data_gen)
    
    # training/validation loops
    for epoch in range(initial_epoch, epochs):
        start_time = time.time()
        
        # adjust lr
        lr = lr_scheduler(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # training
        model.train()
        train_losses = []
        train_total_loss = []
        for step in range(steps_per_epoch):
            
            # generate inputs (and true outputs) and convert them to tensors
            inputs, labels = next(train_gen)
            inputs = [torch.from_numpy(d).to(device).float() for d in inputs]
            labels = [torch.from_numpy(d).to(device).float() for d in labels]

            # run inputs through the model to produce a warped image and flow field
            pred = model(*inputs)

            # calculate total loss
            loss = 0
            loss_list = []
            for i, Loss in enumerate(Losses):
                curr_loss = Loss(labels[i], pred[i]) * Weights[i]
                loss_list.append(curr_loss.item())
                loss += curr_loss
            train_losses.append(loss_list)
            train_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
       # validation
        model.eval()
        PT_inter = PT_union = 0
        MLN_inter = MLN_union = 0
        Survival_time = []
        Survival_label = []
        for valid_image in valid_samples:
            
            # generate inputs (and true outputs) and convert them to tensors
            PET, CT, Seg_PT, Seg_MLN, Label = datagenerators.load_by_name(data_dir, valid_image)
            PET = torch.from_numpy(PET).to(device).float()
            CT = torch.from_numpy(CT).to(device).float()
            Survival_label.append(Label)

            # run inputs through the model to produce a warped image and flow field
            with torch.no_grad():
                pred = model(PET, CT)

            # calculate validation metrics
            Seg_PT_pred = pred[0].detach().cpu().numpy().squeeze()
            _, Seg_PT_pred = cv2.threshold(Seg_PT_pred,0.5,1,cv2.THRESH_BINARY)
            PT_inter = PT_inter + np.sum(Seg_PT_pred * Seg_PT)
            PT_union = PT_union + np.sum(Seg_PT_pred + Seg_PT)
        
            Seg_MLN_pred = pred[1].detach().cpu().numpy().squeeze()
            _, Seg_MLN_pred = cv2.threshold(Seg_MLN_pred,0.5,1,cv2.THRESH_BINARY)
            MLN_inter = MLN_inter + np.sum(Seg_MLN_pred * Seg_MLN)
            MLN_union = MLN_union + np.sum(Seg_MLN_pred + Seg_MLN)
            
            Survival_pred = pred[2].detach().cpu().numpy().squeeze()
            Survival_time.append(Get_survival_time(Survival_pred))
        
        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, epochs)
        time_info = 'Total %.2f sec' % (time.time() - start_time)
        train_losses = ', '.join(['%.4f' % f for f in np.mean(train_losses, axis=0)])
        train_loss_info = 'Train loss: %.4f (%s)' % (np.mean(train_total_loss), train_losses)
        Dice_PT, Dice_MLN = 2*PT_inter/PT_union, 2*MLN_inter/MLN_union
        valid_dice_info = 'Valid DSC: %.4f (%.4f, %.4f)' % (np.mean([Dice_PT, Dice_MLN]), Dice_PT, Dice_MLN)
        valid_cindex = concordance_index(np.array(Survival_label)[:,0], Survival_time, np.array(Survival_label)[:,1])
        valid_cindex_info = 'Valid C-index: %.4f' % valid_cindex
        print(' - '.join((epoch_info, time_info, train_loss_info, valid_dice_info, valid_cindex_info)), flush=True)
    
        # save model checkpoint
        torch.save(model.state_dict(), os.path.join(model_dir, '%02d.pt' % (epoch+1)))

        
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        dest="data_dir", default='./',
                        help="data folder")
    parser.add_argument("--train_samples", type=str,
                        dest="train_samples", default='./',
                        help="training samples")
    parser.add_argument("--valid_samples", type=str,
                        dest="valid_samples", default='./',
                        help="validation samples")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='./models/',
                        help="models folder")
    parser.add_argument("--load_model", type=str,
                        dest="load_model", default='./',
                        help="load model file to initialize with")
    parser.add_argument("--device", type=str, default='gpu0',
                        dest="device", help="gpuN or multi-gpu")
    parser.add_argument("--initial_epoch", type=int,
                        dest="initial_epoch", default=0,
                        help="initial_epoch")
    parser.add_argument("--epochs", type=int,
                        dest="epochs", default=60,
                        help="number of epoch")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=200,
                        help="iterations of each epoch")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=2,
                        help="batch_size")

    args = parser.parse_args()
    train(**vars(args))
