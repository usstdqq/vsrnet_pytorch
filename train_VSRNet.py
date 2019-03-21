from __future__ import print_function
import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import time

from math import log10
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_utils import DatasetFromH5_MFSR
from model import Net_VSRNet
from tensorboard_logger import configure, log_value


# Training settings
parser = argparse.ArgumentParser(description='PyTorch ESPCN')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=256, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=400, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--srcnn_model', type=str, default='epochs_SRCNN/model_epoch_800.pth', help='pre-trained SRCNN model')
opt = parser.parse_args()

print(opt)

print('===> Select GPU to train...') 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets...')
# Please set the path to training and validation data here
# Suggest to put the data in SSD to get better data IO speed
train_set = DatasetFromH5_MFSR('/home/dqq/SR_ZOO/Data/CDVL/uf_4/train/Data_CDVL_LR_Bic_MC_uf_4_ps_72_fn_5_tpn_225000.h5',
                               '/home/dqq/SR_ZOO/Data/CDVL/uf_4/train/Data_CDVL_HR_uf_4_ps_72_fn_5_tpn_225000.h5',
                               upscale_factor=opt.upscale_factor, input_transform=transforms.ToTensor(), target_transform=transforms.ToTensor())

val_set = DatasetFromH5_MFSR('/home/dqq/SR_ZOO/Data/CDVL/uf_4/val/Data_CDVL_LR_Bic_MC_uf_4_ps_72_fn_5_tpn_45000.h5',
                             '/home/dqq/SR_ZOO/Data/CDVL/uf_4/val/Data_CDVL_HR_uf_4_ps_72_fn_5_tpn_45000.h5',
                             upscale_factor=opt.upscale_factor, input_transform=transforms.ToTensor(), target_transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

print('===> Building model...')
model = Net_VSRNet(upscale_factor=opt.upscale_factor, srcnn_model=opt.srcnn_model)
criterion = nn.MSELoss()

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

print('===> Parameters:', sum(param.numel() for param in model.parameters()))

print('===> Initialize Optimizer...')      
#optimizer = optim.Adam(model.parameters(), lr=opt.lr)
optimizer = optim.Adam([{'params': model.conv1_f0.parameters()},
                        {'params': model.conv1_f1.parameters()},
                        {'params': model.conv1_f2.parameters()},
                        {'params': model.conv2.parameters()},
                        {'params': model.conv3.parameters(), 'lr': opt.lr/10.0}
                        ], lr=opt.lr)

print('===> Initialize Logger...')     
configure("tensorBoardRuns/VSRNet-relu-mid-fusion-pretrain-sym-x4-batch-128-CDVL-225000x5x72x72-wd")


def train(epoch):
    epoch_loss = 0
    epoch_psnr = 0
    start = time.time()
    #   Step up learning rate decay
    #   The network have 3 layers
    lr = opt.lr * (0.1 ** (epoch // (opt.nEpochs // 4)))
    
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr
    optimizer.param_groups[2]['lr'] = lr
    optimizer.param_groups[3]['lr'] = lr
    optimizer.param_groups[4]['lr'] = lr/10.0
    
    for iteration, batch in enumerate(train_loader, 1):
        image, target = Variable(batch[0]), Variable(batch[1])
        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        loss = criterion(model(image), target)
        psnr = 10 * log10(1 / loss.data.item())
        epoch_loss += loss.data.item()
        epoch_psnr += psnr
        loss.backward()
        optimizer.step()
        
    end = time.time()
    print("===> Epoch {} Complete: lr: {}, Avg. Loss: {:.4f}, Avg.PSNR:  {:.4f} dB, Time: {:.4f}".format(epoch, lr, epoch_loss / len(train_loader), epoch_psnr / len(train_loader), (end-start)))
    
    log_value('train_loss', epoch_loss / len(train_loader), epoch)
    log_value('train_psnr', epoch_psnr / len(train_loader), epoch)



def val(epoch):
    #   Validation on CDVL val set
    avg_psnr = 0
    avg_mse = 0
    frame_count = 0
    start = time.time()
    for batch in val_loader:
        image, target = Variable(batch[0]), Variable(batch[1])
        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()

        prediction = model(image)
        
        for i in range(0, image.shape[0]):
            mse = criterion(prediction[i], target[i])
            psnr = 10 * log10(1 / mse.data.item())
            avg_psnr += psnr
            avg_mse  += mse.data.item()
            frame_count += 1

    end = time.time()
    print("===> Epoch {} Validation CDVL: Avg. Loss: {:.4f}, Avg.PSNR:  {:.4f} dB, Time: {:.4f}".format(epoch, avg_mse / frame_count, avg_psnr / frame_count, (end-start)))

    log_value('val_loss', avg_mse / frame_count, epoch)
    log_value('val_psnr', avg_psnr / frame_count, epoch)


def checkpoint(epoch):
    if epoch%10 == 0:
        if not os.path.exists("epochs_VSRNet"):
            os.makedirs("epochs_VSRNet")
        model_out_path = "epochs_VSRNet/" + "model_epoch_{}.pth".format(epoch)
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

val(0)
checkpoint(0)
for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    val(epoch)
    checkpoint(epoch)
