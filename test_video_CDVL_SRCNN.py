import argparse
import os
from os import listdir
import h5py
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from data_utils import is_h5_file
from model import Net_SRCNN
from psnr import psnr
from ssim import ssim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Super Resolution')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--model_name', default='epochs_SRCNN/model_epoch_800.pth', type=str, help='SRCNN model name')
    parser.add_argument('--is_real_time', default=False, type=bool, help='super resolution real time to show')
    parser.add_argument('--delay_time', default=1, type=int, help='super resolution delay time to show')
    opt = parser.parse_args()
    
    print(opt)
    
    print('===> Select GPU to test...') 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    UPSCALE_FACTOR = opt.upscale_factor
    IS_REAL_TIME = opt.is_real_time
    DELAY_TIME = opt.delay_time
    MODEL_NAME = opt.model_name
    
    # Path to the testing data: Bicubic upscaled videos and HR groud truth
    path_LR_Bic_MC = '/home/dqq/HDData_2/Dropbox/SR_ZOO/Data/CDVL/uf_' + str(UPSCALE_FACTOR) + '/test/LR_Bic_MC'
    path_HR = '/home/dqq/HDData_2/Dropbox/SR_ZOO/Data/CDVL/uf_' + str(UPSCALE_FACTOR) + '/test/HR'
    videos_h5_name = [x for x in listdir(path_LR_Bic_MC) if is_h5_file(x)]
    videos_h5_name.sort()
    
    #   Prepare to save PSNR and SSIM
    #   Each value corresponding to one test video
    h5_len = len(videos_h5_name)
    model_PSNR   = np.zeros(h5_len)
    model_SSIM   = np.zeros(h5_len)
    bicubic_PSNR = np.zeros(h5_len)
    bicubic_SSIM = np.zeros(h5_len)
    model_time   = np.zeros(h5_len)
    
    
    model = Net_SRCNN(upscale_factor=UPSCALE_FACTOR)
    model = torch.load(MODEL_NAME)
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
    
    out_path = 'results/CDVL/SRCNN_uf_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    
    video_idx = 0
    for video_name in tqdm(videos_h5_name, desc='convert LR videos to HR videos'):
        #   Read h5 file
        LR_Bic_MC_h5_file = h5py.File(path_LR_Bic_MC + '/' +video_name, 'r')
        LR_Bic_MC_h5_data = LR_Bic_MC_h5_file['data']
        HR_h5_file = h5py.File(path_HR + '/' + video_name, 'r')
        HR_h5_data = HR_h5_file['data']
        
        # load to memory
        HR_h5_data = HR_h5_data.value
        LR_Bic_MC_h5_data = LR_Bic_MC_h5_data.value
        
        # transpose to correct order
        HR_h5_data = np.transpose(HR_h5_data, (3, 2, 1, 0))
        LR_Bic_MC_h5_data = np.transpose(LR_Bic_MC_h5_data, (3, 2, 1, 0))
        
        frame_number = LR_Bic_MC_h5_data.shape[0]
        
        if not IS_REAL_TIME:
            fps = 30
            size = (LR_Bic_MC_h5_data.shape[3], LR_Bic_MC_h5_data.shape[2])
            output_name = out_path + video_name.split('.')[0] + '.avi'
            videoWriter = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)
#            videoWriter = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
                
        #   Prepare to save PSNR and SSIM of the current video
        #   Each value corresponding to one test frame
        model_PSNR_cur   = np.zeros(frame_number)
        model_SSIM_cur   = np.zeros(frame_number)
        bicubic_PSNR_cur = np.zeros(frame_number)
        bicubic_SSIM_cur = np.zeros(frame_number)
        model_time_cur   = np.zeros(frame_number)
        
        for idx in tqdm(range(0, frame_number)):
            img_HR = HR_h5_data[idx, 0, :, :] #2D
            img_LR_Bic = LR_Bic_MC_h5_data[idx, 2, :, :] #2D
            
            # Reshape to 4D
            img_LR_Bic = img_LR_Bic.reshape((1, 1, img_LR_Bic.shape[0], img_LR_Bic.shape[1]))
            
            img_LR_Bic = img_LR_Bic.astype(np.float32)
        
            img_LR_Bic =  torch.from_numpy(img_LR_Bic)
                               
            if torch.cuda.is_available():
                img_LR_Bic = img_LR_Bic.cuda()
        
            start = time.time()
            if img_LR_Bic.sum() != 0:
                img_HR_net = model(img_LR_Bic)
            else:
                img_HR_net = img_LR_Bic
                
            end = time.time() # measure the computation time
            
            img_HR_net = img_HR_net.cpu()
            img_HR_net = img_HR_net.data[0].numpy()
            img_HR_net *= 255.0
            img_HR_net = img_HR_net.clip(0, 255)
            img_HR_net = img_HR_net.astype(np.uint8)
            
            img_LR_Bic = img_LR_Bic.cpu()
            img_LR_Bic = img_LR_Bic.data[0].numpy()
            img_LR_Bic *= 255.0
            img_LR_Bic = img_LR_Bic.clip(0, 255)
            img_LR_Bic = img_LR_Bic.astype(np.uint8)
            
            img_HR = img_HR.reshape((1, img_HR.shape[0], img_HR.shape[1]))

            
            model_PSNR_cur[idx]   = psnr((img_HR).reshape(img_HR.shape[1], img_HR.shape[2]).astype(int), (img_HR_net).reshape(img_HR_net.shape[1], img_HR_net.shape[2]).astype(int))
            model_SSIM_cur[idx]   = ssim((img_HR).reshape(img_HR.shape[1], img_HR.shape[2]).astype(int), (img_HR_net).reshape(img_HR_net.shape[1], img_HR_net.shape[2]).astype(int))
            bicubic_PSNR_cur[idx] = psnr((img_HR).reshape(img_HR.shape[1], img_HR.shape[2]).astype(int), (img_LR_Bic).reshape(img_LR_Bic.shape[1], img_LR_Bic.shape[2]).astype(int))
            bicubic_SSIM_cur[idx] = ssim((img_HR).reshape(img_HR.shape[1], img_HR.shape[2]).astype(int), (img_LR_Bic).reshape(img_LR_Bic.shape[1], img_LR_Bic.shape[2]).astype(int))
            model_time_cur[idx]   = (end-start)
        
            # Repeat to 3 channels to save and display
            img_HR_net = np.repeat(img_HR_net, 3, axis=0)
            img_HR_net = np.transpose(img_HR_net, (1, 2, 0))

            if IS_REAL_TIME:
                plt.imshow(img_HR_net, cmap = 'gray')
                plt.show()

#                cv2.imshow('LR Video ', img_LR_Bic)
#                cv2.imshow('SR Video ', img_HR_net)
#                cv2.waitKey(DELAY_TIME)
            else:
                # save video
                videoWriter.write(img_HR_net)
        
        # Done video writing
        videoWriter.release()
        
        # Save PSNR and SSIM
        # Exclude PSNR = 100 cases (caused by black frames)
        cal_flag = (model_PSNR_cur != 100)
        model_PSNR[video_idx]   = np.mean(model_PSNR_cur[cal_flag])
        model_SSIM[video_idx]   = np.mean(model_SSIM_cur[cal_flag])
        bicubic_PSNR[video_idx] = np.mean(bicubic_PSNR_cur[cal_flag])
        bicubic_SSIM[video_idx] = np.mean(bicubic_SSIM_cur[cal_flag])
        model_time[video_idx]   = np.mean(model_time_cur[cal_flag])
        
        print("===> Test on Video Idx: " + str(video_idx) +" Complete: Model PSNR: {:.4f} dB, Model SSIM: {:.4f} , Bicubic PSNR:  {:.4f} dB, Bicubic SSIM: {:.4f} , Average time: {:.4f}"
          .format(model_PSNR[video_idx], model_SSIM[video_idx], bicubic_PSNR[video_idx], bicubic_SSIM[video_idx], model_time[video_idx]*1000))
        video_idx += 1
    
    h5_file_name = out_path + 'CDVL_SRCNN_test_stats.h5'
    with h5py.File(h5_file_name, 'w') as hf:
        hf.create_dataset("model_PSNR",   data=model_PSNR)
        hf.create_dataset("model_SSIM",   data=model_SSIM)
        hf.create_dataset("bicubic_PSNR", data=bicubic_PSNR)
        hf.create_dataset("bicubic_SSIM", data=bicubic_SSIM)
        hf.create_dataset("model_time",   data=model_time)
    print("===> Test on All Videos Complete: Model PSNR: {:.4f} dB, Model SSIM: {:.4f} , Bicubic PSNR:  {:.4f} dB, Bicubic SSIM: {:.4f} , Average time: {:.4f}"
          .format(np.average(model_PSNR), np.average(model_SSIM), np.average(bicubic_PSNR), np.average(bicubic_SSIM), np.average(model_time)*1000))
