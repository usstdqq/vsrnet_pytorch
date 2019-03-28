
# VSRNet
A PyTorch implementation of VSRNet
[Video super-resolution with convolutional neural networks] (https://ieeexplore.ieee.org/abstract/document/7444187/)

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision -c soumith
conda install pytorch torchvision cuda80 -c soumith # install it if you have installed cuda 8.0
```
- PyTorchNet
```
pip install git+https://github.com/pytorch/tnt.git@master
```
- tqdm
```
pip install tqdm
```
- opencv
```
conda install -c conda-forge opencv
```
- tensorboard_logger
```
pip install tensorboard_logger
```
- h5py
```
conda install h5py
```
- [pyflow](https://github.com/pathak22/pyflow)


## Datasets

### Train, Val, Test Video Dataset
The train and val datasets are sampled from [CDVL dataset](https://www.cdvl.org/about/index.php).
We choose this dataset because we want to extend single frame based SRCNN to multi frame based VSRNet
Train dataset (uf_X, X=2,3,4) is composed of multiple h5 files:
- Data_CDVL_LR_uf_4_ps_72_fn_5_tpn_225000.h5: patches sampled from LR frames, 225243x5x(72/X)x(72/X)
- Data_CDVL_HR_uf_4_ps_72_fn_5_tpn_225000.h5: patches sampled from HR frames, 225243x5x72x72
- Data_CDVL_LR_Bic_uf_4_ps_72_fn_5_tpn_225000.h5:  patches sampled from Matlab bicubic interpolation upscaled frames, 225243x5x72x72
- Data_CDVL_LR_Bic_MC_uf_4_ps_72_fn_5_tpn_225000.h5:  patches sampled from Matlab bicubic interpolation upscaled and optical flow motion compensated frames, 225243x5x72x72

Val dataset (uf_X, X=2,3,4) is composed of multiple h5 files:
- Data_CDVL_LR_uf_4_ps_72_fn_5_tpn_45000.h5: patches sampled from LR frames, 45159x5x(72/X)x(72/X)
- Data_CDVL_HR_uf_4_ps_72_fn_5_tpn_45000.h5: patches sampled from HR frames, 45159x5x72x72
- Data_CDVL_LR_Bic_uf_4_ps_72_fn_5_tpn_45000.h5:  patches sampled from Matlab bicubic interpolation upscaled frames, 45159x5x72x72
- Data_CDVL_LR_Bic_MC_uf_4_ps_72_fn_5_tpn_45000.h5:  patches sampled from Matlab bicubic interpolation upscaled and optical flow motion compensated frames, 45159x5x72x72

Test dataset (uf_X, X=2,3,4) is composed of multiple h5 folders:
- LR: LR frames of a scene, (1920/X)x(1080/X)x5xframe_number
- HR: HR frames of a scene, 1920x1080x1xframe_number
- LR_Bic: Matlab Bicubic upscaled LR frames of a scene, 1920x1080x5xframe_number
- LR_Bic_MC: Matlab Bicubic upscaled and [Celiu optical flow](https://people.csail.mit.edu/celiu/OpticalFlow/) motion compensated LR frames of a scene, 1920x1080x5xframe_number
- LR_MC: [Celiu optical flow](https://people.csail.mit.edu/celiu/OpticalFlow/) motion compensated LR frames of a scene, (1920/X)x(1080/X)x5xframe_number

Download the pre-processed h5 files from [here](https://www.dropbox.com/sh/1jz9zeer9wxetx2/AACKqSzh15QPNjyp7Nq_g77_a?dl=0), 
and then setup the path in the codes.

### Test Image Dataset
The test image dataset are sampled from 
| **Set 5** |  [Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)
| **Set 14** |  [Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests)
| **BSD 100** | [Martin et al. ICCV 2001](https://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
| **Sun-Hays 80** | [Sun and Hays ICCP 2012](http://cs.brown.edu/~lbsun/SRproj2012/SR_iccp2012.html)
| **Urban 100** | [Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr).
Download the preprocessed H5 files from [here](https://www.dropbox.com/sh/2ozntfm5i9y9h9c/AABYHwsOSIBgn1XkhDsSIIjca?dl=0), and then setup the path in test_image.py file.



## Usage

### Train SRCNN 
SRCNN need to be trained to initialize VSRNet

```
python train_SRCNN.py

optional arguments:
--upscale_factor      super resolution upscale factor [default value is 4]
--num_epochs          super resolution epochs number [default value is 800]
```

### Train VSRNet 

```
python train_VSRNet.py

optional arguments:
--upscale_factor      super resolution upscale factor [default value is 4]
--num_epochs          super resolution epochs number [default value is 400]
```

### Test Image
```
python test_image_SRCNN.py

optional arguments:
--upscale_factor      super resolution upscale factor [default value is 3]
--model_name          super resolution model name [default value is epochs_SRCNN/model_epoch_800.pth]
```
The output high resolution images are on `results` directory.

### Test CDVL Video (preprocessed H5 files) using SRCNN
```
python test_video_CDVL_SRCNN.py

optional arguments:
--upscale_factor      super resolution upscale factor [default value is 3]
--model_name          SRCNN model name [default value is epochs_SRCNN/model_epoch_800.pth]
--is_real_time        whether to save results into video file or not [default value is False]
--delay_time          display delay time [default value is 1]
```
The output high resolution images are on `results` directory.

### Test CDVL Video (preprocessed H5 files) using VSRNet
```
python test_video_CDVL_VSRNet.py

optional arguments:
--upscale_factor      super resolution upscale factor [default value is 3]
--vsrnet_model_name   VSRNet model name [default value is epochs_SRCNN/model_epoch_800.pth]
--srcnn_model_name    SRCNN model name [default value is epochs_SRCNN/model_epoch_800.pth]
--is_real_time        whether to save results into video file or not [default value is False]
--delay_time          display delay time [default value is 1]
```
The output high resolution images are on `results` directory.


