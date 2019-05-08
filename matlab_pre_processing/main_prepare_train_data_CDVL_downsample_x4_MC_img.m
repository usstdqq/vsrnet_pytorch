clear;
clc;
close all;

%%  Set up parameters for training
dataFolder = '../CDVL_Processed/train/';
% total_train_patch_number = 225000;
% total_val_patch_number = 45000;
frame_number = 5;
frame_sample_rate = 2;
patch_size = 72;
upscale_factor = 4;

frameHeight_HR = 1080;
frameWidth_HR = 1920;

frameHeight_LR = frameHeight_HR/upscale_factor;
frameWidth_LR = frameWidth_HR/upscale_factor;

%%  Get the video names 
videoList = dir([dataFolder '*.avi']);
videosLen = length(videoList);
total_frame_number = 0;
video_frame_number = zeros(videosLen, 1);
patch_size_LR = patch_size/upscale_factor;

for vid_idx = 1 : videosLen
    vid_idx
    %   Load the whole video
    v = VideoReader([dataFolder, videoList(vid_idx).name]);
    curFrameNumber = v.NumberOfFrames ;
    frameCount = floor(floor((curFrameNumber - 2) / frame_sample_rate) / frame_number) * frame_number;
    
    if frameCount < 2 * frame_number
        frameCount = 2 * frameCount;
    end
    
    buf = zeros(frameCount, frameHeight_HR, frameWidth_HR, 'uint8');
    bufLR = zeros(frameCount, frameHeight_LR, frameWidth_LR, 'uint8');
    bufLR_Bic = zeros(frameCount, frameHeight_HR, frameWidth_HR, 'single');
    bufLR_Bic_MC = zeros(frameCount, frameHeight_HR, frameWidth_HR, 'single');
    bufLR_MC = zeros(frameCount, frameHeight_LR, frameWidth_LR, 'single');
    
    for j = 1 : frameCount
        
        cur_frame_rgb = read(v,j);
        cur_frame_ycbcr = rgb2ycbcr(cur_frame_rgb);
        cur_frame_y = cur_frame_ycbcr(:,:,1);
        
%         cur_frame_y = imresize(cur_frame_y, [frameHeight, frameWidth], 'bicubic');
        
        cur_frame_y_LR = imresize(cur_frame_y, [frameHeight_LR, frameWidth_LR], 'bicubic');
        cur_frame_y_LR_Bic = imresize(single(cur_frame_y_LR)/255.0, [frameHeight_HR, frameWidth_HR], 'bicubic');
        
        
        buf(j,:,:) = cur_frame_y;
        bufLR(j,:,:) = cur_frame_y_LR;
        bufLR_Bic(j,:,:) = cur_frame_y_LR_Bic;
        
    end
    
    %   Perform motion compensation
    
    % set optical flow parameters (see Coarse2FineTwoFrames.m for the definition of the parameters)
    alpha = 0.012;
    ratio = 0.75;
    minWidth = 20;
    nOuterFPIterations = 7;
    nInnerFPIterations = 1;
    nSORIterations = 30;

    para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];
    half_frame_number = (frame_number - 1) / 2;
    
    for fc = 1 : frameCount
%     for fc = 1 : 5    
        cur_sec = floor((fc-1)/frame_number);
        fc_base = cur_sec * frame_number + half_frame_number + 1;
        sprintf('cur_sec: %d, fc: %d, fc_base: %d', cur_sec, fc, fc_base);
        if fc ~= fc_base
            im1 = squeeze(bufLR_Bic(fc_base, :, :));
            im2 = squeeze(bufLR_Bic(fc, :, :));
            
            im1 = single(im1);
            im2 = single(im2);
            
            tic;
            [vx,vy,warpim2] = Coarse2FineTwoFrames(im1,im2,para);
            toc;
             
            bufLR_Bic_MC(fc,:,:) = single(warpim2);
            
            im1 = squeeze(bufLR(fc_base, :, :));
            im2 = squeeze(bufLR(fc, :, :));
            
            im1 = single(im1)/255.0;
            im2 = single(im2)/255.0;
            
            tic;
            [vx,vy,warpim2] = Coarse2FineTwoFrames(im1,im2,para);
            toc;
             
            bufLR_MC(fc,:,:) = single(warpim2);
            
        else
            im1 = squeeze(bufLR_Bic(fc, :, :));
            im1 = single(im1);
            bufLR_Bic_MC(fc,:,:) = im1;
            
            im1 = squeeze(bufLR(fc, :, :));
            im1 = single(im1)/255.0;
            bufLR_MC(fc,:,:) = im1;
            
        end
        
    end
    

    %%  Save to h5 file
    buf_name = ['CDVL_Train_Data_img_df_', num2str(upscale_factor),'/HR/scene_', num2str(vid_idx), '.h5'];
    hdf5write(buf_name, '/data', buf, 'V71Dimensions', false);
    
    bufLR_Bic_name = ['CDVL_Train_Data_img_df_', num2str(upscale_factor), '/LR_Bic/scene_', num2str(vid_idx), '.h5'];
    hdf5write(bufLR_Bic_name, '/data', bufLR_Bic, 'V71Dimensions', false);
    
    bufLR_name = ['CDVL_Train_Data_img_df_', num2str(upscale_factor), '/LR/scene_', num2str(vid_idx), '.h5'];
    hdf5write(bufLR_name, '/data', bufLR, 'V71Dimensions', false);
    
    bufLR_Bic_MC_name = ['CDVL_Train_Data_img_df_', num2str(upscale_factor), '/LR_Bic_MC/scene_', num2str(vid_idx), '.h5'];
    hdf5write(bufLR_Bic_MC_name, '/data', bufLR_Bic_MC, 'V71Dimensions', false);
%     
    bufLR_MC_name = ['CDVL_Train_Data_img_df_', num2str(upscale_factor), '/LR_MC/scene_', num2str(vid_idx), '.h5'];
    hdf5write(bufLR_MC_name, '/data', bufLR_MC, 'V71Dimensions', false);

    
end



