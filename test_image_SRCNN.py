import argparse
import cv2
import os
import h5py
import torch
import numpy as np
from os import listdir
import time

from torch.autograd import Variable
from tqdm import tqdm
from data_utils import is_image_file
from model import Net_SRCNN
from psnr import psnr
from ssim import ssim

#from PIL import Image
#from torchvision.transforms import ToTensor

def rgb2ycbcr(im_rgb): #im_rgb in [0, 1]
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
    return im_ycbcr

def ycbcr2rgb(im_ycbcr): #im_ycbcr in [0, 1]
    im_ycbcr = im_ycbcr.astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*255.0-16)/(235-16) #to [0, 1]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*255.0-16)/(240-16) #to [0, 1]
    im_ycrcb = im_ycbcr[:,:,(0,2,1)].astype(np.float32)
    im_rgb = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCR_CB2RGB)
    return im_rgb

def rgb2bgr(im_rgb): #im_ycbcr in [0, 1]
    im_bgr = im_rgb[:,:,(2,1,0)]
    return im_bgr

def is_h5_file(filename):
    return any(filename.endswith(extension) for extension in ['.h5'])

def testFromH5Folder(input_dataset_dir, output_dataset_dir, upscale_factor, input_field_name, model_name, cuda_flag, if_save):
    
    h5_file_list = [x for x in listdir(input_dataset_dir) if is_h5_file(x)]
    h5_file_list.sort()
    h5_len = len(h5_file_list)
    
    model_PSNR = np.zeros(h5_len)
    model_SSIM = np.zeros(h5_len)
    bicubic_PSNR = np.zeros(h5_len)
    bicubic_SSIM = np.zeros(h5_len)
    
    model_time = np.zeros(h5_len)
    
    #   Load SR Model
    model = Net_SRCNN(upscale_factor=UPSCALE_FACTOR)
    model = torch.load(MODEL_NAME)
    if torch.cuda.is_available() & cuda_flag:
        model = model.cuda()
    else:
        model = model.cpu()
    
    
    if not os.path.exists(output_dataset_dir):
        os.makedirs(output_dataset_dir)
    
    for idx in tqdm(range(h5_len), desc='SR of upscale factor = ' + str(upscale_factor)):
        h5_file_name = h5_file_list[idx]
        h5_file = h5py.File(input_dataset_dir + '/' + h5_file_name, 'r')
        
        img_HR_y      = h5_file['HR'].value
        img_LR_y      = h5_file['LR'].value
        img_LR_bic_y  = h5_file['LR_bic_y'].value
        img_LR_bic_cb = h5_file['LR_bic_cb'].value
        img_LR_bic_cr = h5_file['LR_bic_cr'].value
        img_HR_RGB    = h5_file['HR_RGB'].value
        
        
        
        img_HR_y      = img_HR_y.astype(np.float32)
        img_LR_y      = img_LR_y.astype(np.float32)
        img_LR_bic_y  = img_LR_bic_y.astype(np.float32)
        img_LR_bic_cb = img_LR_bic_cb.astype(np.float32)
        img_LR_bic_cr = img_LR_bic_cr.astype(np.float32)
        img_HR_RGB    = img_HR_RGB.astype(np.float32)
        
        img_HR_y   = img_HR_y / 255.0
        img_LR_y   = img_LR_y / 255.0
        img_HR_RGB = img_HR_RGB / 255.0
        
        if input_field_name == 'LR':
            img_LR_4d = img_LR_y.reshape(1, 1, img_LR_y.shape[0], img_LR_y.shape[1])
        elif input_field_name == 'LR_bic_y':
            img_LR_4d = img_LR_bic_y.reshape(1, 1, img_LR_bic_y.shape[0], img_LR_bic_y.shape[1])
        
        
        image = Variable(torch.from_numpy(img_LR_4d))
        
        if torch.cuda.is_available() & cuda_flag:
            image = image.cuda()
            
            
        start = time.time()
        target = model(image)
        end = time.time()
        
        target = target.cpu()
        
        img_HR_y_net = target.data[0][0].numpy()
        
        
        if if_save:
            img_HR_ycbcr_net = np.zeros(img_HR_RGB.shape)
            img_HR_ycbcr_net[:,:,0] = img_HR_y_net
            img_HR_ycbcr_net[:,:,1] = img_LR_bic_cb
            img_HR_ycbcr_net[:,:,2] = img_LR_bic_cr
            
            img_HR_ycbcr_net = img_HR_ycbcr_net.clip(0.0, 1.0)
            
            img_HR_RGB_net = ycbcr2rgb(img_HR_ycbcr_net)
            img_HR_RGB_net = img_HR_RGB_net.clip(0.0, 1.0)
            img_HR_RGB_net *= 255.0
            
            img_HR_BGR_net = rgb2bgr(img_HR_RGB_net)
            
            image_name = 'img_' + str(idx) + '_net.png'
            cv2.imwrite(output_dataset_dir + '/' + image_name, img_HR_BGR_net.astype(np.uint8))
        
        
        #   Compute Stat
        model_PSNR[idx]   = psnr((img_HR_y*255.0).astype(int), (img_HR_y_net*255.0).astype(int))
        model_SSIM[idx]   = ssim((img_HR_y*255.0).astype(int), (img_HR_y_net*255.0).astype(int))
        bicubic_PSNR[idx] = psnr((img_HR_y*255.0).astype(int), (img_LR_bic_y*255.0).astype(int))
        bicubic_SSIM[idx] = ssim((img_HR_y*255.0).astype(int), (img_LR_bic_y*255.0).astype(int))
        model_time[idx]   = (end-start)
        
    print("===> Test on" + input_dataset_dir +" Complete: Model PSNR: {:.4f} dB, Model SSIM: {:.4f} , Bicubic PSNR:  {:.4f} dB, Bicubic SSIM: {:.4f} , Average time: {:.4f}"
          .format(np.average(model_PSNR), np.average(model_SSIM), np.average(bicubic_PSNR), np.average(bicubic_SSIM), np.average(model_time)*1000))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Super Resolution')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--model_name', default='epochs_SRCNN/model_epoch_800.pth', type=str, help='super resolution model name')
    parser.add_argument('--if_save', default=True, type=bool, help='whether or not to save the super-resolved images')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name
    IF_SAVE = opt.if_save

##  ================================================================================   
##  Test Set5
    input_path = '../../../Data/Set5_SR/h5_SRF_4/'
    output_path = 'results/Set5_SR'
    testFromH5Folder(input_path, output_path, UPSCALE_FACTOR, 'LR_bic_y', MODEL_NAME, 0, IF_SAVE)
    testFromH5Folder(input_path, output_path, UPSCALE_FACTOR, 'LR_bic_y', MODEL_NAME, 1, IF_SAVE)

##  ================================================================================   
##  Test Set14
    input_path = '../../../Data/Set14_SR/h5_SRF_4/'
    output_path = 'results/Set14_SR'
    testFromH5Folder(input_path, output_path, UPSCALE_FACTOR, 'LR_bic_y', MODEL_NAME, 0, IF_SAVE)
    testFromH5Folder(input_path, output_path, UPSCALE_FACTOR, 'LR_bic_y', MODEL_NAME, 1, IF_SAVE)

##  ================================================================================   
##  Test Urban100
    input_path = '../../../Data/Urban100_SR/h5_SRF_4/'
    output_path = 'results/Urban100_SR'
    testFromH5Folder(input_path, output_path, UPSCALE_FACTOR, 'LR_bic_y', MODEL_NAME, 0, IF_SAVE)
    testFromH5Folder(input_path, output_path, UPSCALE_FACTOR, 'LR_bic_y', MODEL_NAME, 1, IF_SAVE)

##  ================================================================================   
##  Test BSD100
    input_path = '../../../Data/BSD100_SR/h5_SRF_4/'
    output_path = 'results/BSD100_SR'
    testFromH5Folder(input_path, output_path, UPSCALE_FACTOR, 'LR_bic_y', MODEL_NAME, 0, IF_SAVE)
    testFromH5Folder(input_path, output_path, UPSCALE_FACTOR, 'LR_bic_y', MODEL_NAME, 1, IF_SAVE)

##  ================================================================================   
##  Test SunHays80
    input_path = '../../../Data/SunHays80_SR/h5_SRF_4/'
    output_path = 'results/SunHays80_SR'
    testFromH5Folder(input_path, output_path, UPSCALE_FACTOR, 'LR_bic_y', MODEL_NAME, 0, IF_SAVE)
    testFromH5Folder(input_path, output_path, UPSCALE_FACTOR, 'LR_bic_y', MODEL_NAME, 1, IF_SAVE)
    

    

    

    
