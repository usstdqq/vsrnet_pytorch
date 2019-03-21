import argparse
import os
import h5py
from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
import torch
from torchvision.transforms import Compose, CenterCrop, Scale
from tqdm import tqdm

import cv2
import numpy as np

import time



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'])


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in ['.mp4', '.avi', '.mpg', '.mkv', '.wmv', '.flv'])

def is_h5_file(filename):
    return any(filename.endswith(extension) for extension in ['.h5'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Scale(crop_size // upscale_factor, interpolation=Image.BICUBIC),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size)
    ])

class DatasetFromH5_SFSR(Dataset):
    def __init__(self, image_dataset_dir, target_dataset_dir, upscale_factor, input_transform=None, target_transform=None):
        super(DatasetFromH5_SFSR, self).__init__()
        
        image_h5_file = h5py.File(image_dataset_dir, 'r')
        target_h5_file = h5py.File(target_dataset_dir, 'r')
        image_dataset = image_h5_file['data']
        target_dataset = target_h5_file['data']
        
        self.image_datasets = image_dataset
        self.target_datasets = target_dataset
        self.total_count = image_dataset.shape[0]
        
        self.input_transform = input_transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        image = self.image_datasets[index, [2], :, :]
        target = self.target_datasets[index, [2], :, :]
        
        image  = image.astype(np.float32)
        target = target.astype(np.float32)
        
        #   Notice that image is the bicubic upscaled LR image patch, in float format, in range [0, 1]
#        image = image / 255.0 
        #   Notice that target is the HR image patch, in uint8 format, in range [0, 255]
        target = target / 255.0
        
        image =  torch.from_numpy(image)
        target = torch.from_numpy(target)

        return image, target

    def __len__(self):
        return self.total_count
    
class DatasetFromH5_MFSR(Dataset):
    def __init__(self, image_dataset_dir, target_dataset_dir, upscale_factor, input_transform=None, target_transform=None):
        super(DatasetFromH5_MFSR, self).__init__()
        
        image_h5_file = h5py.File(image_dataset_dir, 'r')
        target_h5_file = h5py.File(target_dataset_dir, 'r')
        image_dataset = image_h5_file['data']
        target_dataset = target_h5_file['data']
        
        self.image_datasets = image_dataset
        self.target_datasets = target_dataset
        self.total_count = image_dataset.shape[0]
        
        self.input_transform = input_transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):        
        image = self.image_datasets[index, :, :, :]
        target = self.target_datasets[index, [2], :, :]
        
        image  = image.astype(np.float32)
        target = target.astype(np.float32)
        
        #   Notice that image is the bicubic upscaled LR image patch, in float format, in range [0, 1]
#        image = image / 255.0 
        #   Notice that target is the HR image patch, in uint8 format, in range [0, 255]
        target = target / 255.0
        
        image =  torch.from_numpy(image)
        target = torch.from_numpy(target)

        return image, target

    def __len__(self):
        return self.total_count
    
    


#   Code to generate the training/validation dataset
#   Because the original CDVL dataset is huge, please use the generated h5 file
def generate_dataset(upscale_factor, patch_size, frame_number, total_patch_number):
    h5_HR_path        = '/home/dqq/HDData3/Data/CDVL_MatlabDataProcess/CDVL_Train_Data_img_df_' + str(upscale_factor) + '/HR'
    h5_LR_path        = '/home/dqq/HDData3/Data/CDVL_MatlabDataProcess/CDVL_Train_Data_img_df_' + str(upscale_factor) + '/LR'
    h5_LR_Bic_path    = '/home/dqq/HDData3/Data/CDVL_MatlabDataProcess/CDVL_Train_Data_img_df_' + str(upscale_factor) + '/LR_Bic'
    h5_LR_Bic_MC_path = '/home/dqq/HDData3/Data/CDVL_MatlabDataProcess/CDVL_Train_Data_img_df_' + str(upscale_factor) + '/LR_Bic_MC'
    h5_LR_MC_path     = '/home/dqq/HDData3/Data/CDVL_MatlabDataProcess/CDVL_Train_Data_img_df_' + str(upscale_factor) + '/LR_MC'
    
    h5_HR_list        = [x for x in listdir(h5_HR_path)        if is_h5_file(x)]
    h5_LR_list        = [x for x in listdir(h5_LR_path)        if is_h5_file(x)]
    h5_LR_Bic_list    = [x for x in listdir(h5_LR_Bic_path)    if is_h5_file(x)]
    h5_LR_Bic_MC_list = [x for x in listdir(h5_LR_Bic_MC_path) if is_h5_file(x)]
    h5_LR_MC_list     = [x for x in listdir(h5_LR_MC_path)     if is_h5_file(x)]
    
    h5_HR_list.sort()
    h5_LR_list.sort()
    h5_LR_Bic_list.sort()
    h5_LR_Bic_MC_list.sort()
    h5_LR_MC_list.sort()
    
    h5_len = len(h5_HR_list)

    root = '/home/dqq/Data/SR_ZOO/data/'
    if not os.path.exists(root):
        os.makedirs(root)
    path = root + '/SRF_' + str(upscale_factor)
    if not os.path.exists(path):
        os.makedirs(path)
        
    #   Get total number of frames
    total_frame_number = 0
    video_frame_number =  np.zeros(h5_len, dtype=int)
    patch_sample_number =  np.zeros(h5_len, dtype=int)
    for vid_idx in tqdm(range(h5_len), desc='compute total frame number of dataset with upscale factor = '
            + str(upscale_factor) + ' from CDVL_MatlabDataProcess'):
        h5_HR_name = h5_HR_list[vid_idx]
        h5_HR_file = h5py.File(h5_HR_path + '/' + h5_HR_name, 'r')
        HR_dataset = h5_HR_file['data']
        
        frameCount = HR_dataset.shape[2]
        frameCount = frameCount // frame_number * frame_number   # round to N * frame_number
        total_frame_number += frameCount
        video_frame_number[vid_idx] = frameCount
        
    #   Get sample patch number per video    
    for vid_idx in tqdm(range(h5_len), desc='compute patch sample number per frame of dataset with upscale factor = '
            + str(upscale_factor) + ' from CDVL_MatlabDataProcess'):
        patch_sample_number[vid_idx] = int(np.ceil(float(video_frame_number[vid_idx]) / float(total_frame_number) * float(total_patch_number)))   
        
    #   Sample pathes (volumes) from Videos
    patch_size_LR = patch_size // upscale_factor
    global_patch_idx = 0
    patchesHR = np.empty((patch_sample_number.sum(), frame_number, patch_size, patch_size), np.dtype('uint8'))
    patchesLR = np.empty((patch_sample_number.sum(), frame_number, int(patch_size//upscale_factor), int(patch_size//upscale_factor)), np.dtype('uint8'))
    patchesLR_Bic = np.empty((patch_sample_number.sum(), frame_number, patch_size, patch_size), np.dtype('float32'))
    patchesLR_Bic_MC = np.empty((patch_sample_number.sum(), frame_number, patch_size, patch_size), np.dtype('float32'))
    patchesLR_MC = np.empty((patch_sample_number.sum(), frame_number, int(patch_size//upscale_factor), int(patch_size//upscale_factor)), np.dtype('float32'))
    
    for vid_idx in tqdm(range(h5_len), desc='generate dataset with upscale factor = '
            + str(upscale_factor) + ' from CDVL_Processed'):
        
        h5_HR_name        = h5_HR_list[vid_idx]
        h5_LR_name        = h5_LR_list[vid_idx]
        h5_LR_Bic_name    = h5_LR_Bic_list[vid_idx]
        h5_LR_Bic_MC_name = h5_LR_Bic_MC_list[vid_idx]
        h5_LR_MC_name     = h5_LR_MC_list[vid_idx]
        
        print('HR h5 name: '+ h5_HR_name + '\n')
        
        h5_HR_file = h5py.File(h5_HR_path + '/' + h5_HR_name, 'r')
        buf_HR = h5_HR_file['data'].value
        buf_HR = np.transpose(buf_HR, (2, 1, 0))
        buf_HR = buf_HR.astype(np.dtype('uint8'))
        
        h5_LR_file = h5py.File(h5_LR_path + '/' + h5_LR_name, 'r')
        bufLR = h5_LR_file['data'].value
        bufLR = np.transpose(bufLR, (2, 1, 0))
        bufLR = bufLR.astype(np.dtype('uint8'))
        
        h5_LR_bic_file = h5py.File(h5_LR_Bic_path + '/' + h5_LR_Bic_name, 'r')
        bufLR_Bic = h5_LR_bic_file['data'].value
        bufLR_Bic = np.transpose(bufLR_Bic, (2, 1, 0))
        bufLR_Bic = bufLR_Bic.astype(np.dtype('float32'))
        
        h5_LR_Bic_MC_file = h5py.File(h5_LR_Bic_MC_path + '/' + h5_LR_Bic_MC_name, 'r')
        bufLR_Bic_MC = h5_LR_Bic_MC_file['data'].value
        bufLR_Bic_MC = bufLR_Bic_MC.astype(np.dtype('float32'))
        
        h5_LR_MC_file = h5py.File(h5_LR_MC_path + '/' + h5_LR_MC_name, 'r')
        bufLR_MC = h5_LR_MC_file['data'].value
        bufLR_MC = bufLR_MC.astype(np.dtype('float32'))
        
        frameCount  = buf_HR.shape[0]
        frameCount = frameCount // frame_number * frame_number   # round to N * frame_number

        frameHeight = buf_HR.shape[1]
        frameWidth  = buf_HR.shape[2]
        
#        center_frame_number = half_frame_number # 0 1 [2] 3 4 
        
        h = np.linspace(0, frameHeight//upscale_factor//patch_size_LR*patch_size_LR-patch_size_LR, frameHeight//upscale_factor//patch_size_LR)
        w = np.linspace(0, frameWidth//upscale_factor//patch_size_LR*patch_size_LR-patch_size_LR, frameWidth//upscale_factor//patch_size_LR)
        f = np.linspace(0, frameCount -frame_number,  frameCount //frame_number)
        
        fv, hv, wv = np.meshgrid(f, h, w)
        fv = np.reshape(fv,(1,-1))
        hv = np.reshape(hv,(1,-1))
        wv = np.reshape(wv,(1,-1))
        
        random_idx = np.random.choice(fv.shape[1], patch_sample_number[vid_idx])
        rand_frame_idx    = fv[0, random_idx]
        rand_frame_height = hv[0, random_idx]
        rand_frame_width  = wv[0, random_idx]
        
        rand_frame_idx    = rand_frame_idx.astype(int)
        rand_frame_height = rand_frame_height.astype(int)
        rand_frame_width  = rand_frame_width.astype(int)
        
        #   Sample patches (volumes) from this video
        for patch_idx in range(0, len(rand_frame_idx)):
            cur_patch_volume_LR = bufLR[rand_frame_idx[patch_idx]:rand_frame_idx[patch_idx]+frame_number, \
                                        rand_frame_height[patch_idx]:rand_frame_height[patch_idx]+patch_size//upscale_factor, \
                                        rand_frame_width[patch_idx] :rand_frame_width[patch_idx]+patch_size//upscale_factor]
           
            #   Note: cur_patch_volume_LR_MC is float type
            cur_patch_volume_LR_MC = bufLR_MC[rand_frame_idx[patch_idx]:rand_frame_idx[patch_idx]+frame_number, \
                                              rand_frame_height[patch_idx]:rand_frame_height[patch_idx]+patch_size//upscale_factor, \
                                              rand_frame_width[patch_idx] :rand_frame_width[patch_idx]+patch_size//upscale_factor]
            
            cur_patch_volume_HR = buf_HR[rand_frame_idx[patch_idx]:rand_frame_idx[patch_idx]+frame_number, \
                                         rand_frame_height[patch_idx]*upscale_factor:rand_frame_height[patch_idx]*upscale_factor+patch_size, \
                                         rand_frame_width[patch_idx] *upscale_factor:rand_frame_width[patch_idx] *upscale_factor+patch_size]
            
            cur_patch_volume_LR_Bic = bufLR_Bic[rand_frame_idx[patch_idx]:rand_frame_idx[patch_idx]+frame_number, \
                                                rand_frame_height[patch_idx]*upscale_factor:rand_frame_height[patch_idx]*upscale_factor+patch_size, \
                                                rand_frame_width[patch_idx] *upscale_factor:rand_frame_width[patch_idx] *upscale_factor+patch_size]
            
            #   Note: cur_patch_volume_LR_Bic_MC is float type
            cur_patch_volume_LR_Bic_MC = bufLR_Bic_MC[rand_frame_idx[patch_idx]:rand_frame_idx[patch_idx]+frame_number, \
                                                      rand_frame_height[patch_idx]*upscale_factor:rand_frame_height[patch_idx]*upscale_factor+patch_size, \
                                                      rand_frame_width[patch_idx] *upscale_factor:rand_frame_width[patch_idx] *upscale_factor+patch_size]
            
            
            patchesLR[global_patch_idx, :, :, :] = cur_patch_volume_LR.copy()
            patchesHR[global_patch_idx, :, :, :] = cur_patch_volume_HR.copy()
            patchesLR_Bic[global_patch_idx, :, :, :] = cur_patch_volume_LR_Bic.copy()
            patchesLR_MC[global_patch_idx, :, :, :] = cur_patch_volume_LR_MC.copy()
            patchesLR_Bic_MC[global_patch_idx, :, :, :] = cur_patch_volume_LR_Bic_MC.copy()
            
            global_patch_idx += 1
            
        
    #   Save to H5 file     
    h5_file_name = 'Data_CDVL_LR_uf_' + str(upscale_factor) + '_ps_' + str(patch_size) + '_fn_' + str(frame_number) + '_tpn_' + str(total_patch_number) + '.h5'
    with h5py.File(h5_file_name, 'w') as hf:
        hf.create_dataset("data",  data=patchesLR)
    
    h5_file_name = 'Data_CDVL_HR_uf_' + str(upscale_factor) + '_ps_' + str(patch_size) + '_fn_' + str(frame_number) + '_tpn_' + str(total_patch_number) + '.h5'
    with h5py.File(h5_file_name, 'w') as hf:
        hf.create_dataset("data",  data=patchesHR)

    h5_file_name = 'Data_CDVL_LR_Bic_uf_' + str(upscale_factor) + '_ps_' + str(patch_size) + '_fn_' + str(frame_number) + '_tpn_' + str(total_patch_number) + '.h5'
    with h5py.File(h5_file_name, 'w') as hf:
        hf.create_dataset("data",  data=patchesLR_Bic)
        
    h5_file_name = 'Data_CDVL_LR_Bic_MC_uf_' + str(upscale_factor) + '_ps_' + str(patch_size) + '_fn_' + str(frame_number) + '_tpn_' + str(total_patch_number) + '.h5'
    with h5py.File(h5_file_name, 'w') as hf:
        hf.create_dataset("data",  data=patchesLR_Bic_MC)
        
    h5_file_name = 'Data_CDVL_LR_MC_uf_' + str(upscale_factor) + '_ps_' + str(patch_size) + '_fn_' + str(frame_number) + '_tpn_' + str(total_patch_number) + '.h5'
    with h5py.File(h5_file_name, 'w') as hf:
        hf.create_dataset("data",  data=patchesLR_MC)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Super Resolution Dataset')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--patch_size',     default=72, type=int, help='patch size for training')
    parser.add_argument('--total_train_patch_number', default=225000, type=int, help='number of training patches')
    parser.add_argument('--total_val_patch_number',   default=45000, type=int, help='number of validation patches')
    parser.add_argument('--frame_number',   default=5, type=int, help='number of temporal frames')
    
    opt = parser.parse_args()
    UPSCALE_FACTOR = opt.upscale_factor
    PATCH_SIZE = opt.patch_size
    TOTAL_TRAIN_PATCH_NUMBER = opt.total_train_patch_number
    TOTAL_VAL_PATCH_NUMBER = opt.total_val_patch_number
    FRAME_NUMBER = opt.frame_number
    

    generate_dataset(upscale_factor=UPSCALE_FACTOR, patch_size=PATCH_SIZE, total_patch_number = TOTAL_TRAIN_PATCH_NUMBER, frame_number = FRAME_NUMBER)
#    generate_dataset(upscale_factor=UPSCALE_FACTOR, patch_size=PATCH_SIZE, total_patch_number = TOTAL_VAL_PATCH_NUMBER, frame_number = FRAME_NUMBER)
