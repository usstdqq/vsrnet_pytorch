import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

class Net_SRCNN(nn.Module):
    def __init__(self, upscale_factor):
        super(Net_SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(1,  64, (9, 9), (1, 1), (4, 4))
        self.conv2 = nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2))
        self.conv3 = nn.Conv2d(32, 1,  (5, 5), (1, 1), (2, 2))
        
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = (self.conv3(x))
        return x
    
    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight)


class Net_VSRNet(nn.Module):
    def __init__(self, upscale_factor, srcnn_model):
        super(Net_VSRNet, self).__init__()

        self.conv1_f0 = nn.Conv2d(1,  64, (9, 9), (1, 1), (4, 4))
        self.conv1_f1 = nn.Conv2d(1,  64, (9, 9), (1, 1), (4, 4))
        self.conv1_f2 = nn.Conv2d(1,  64, (9, 9), (1, 1), (4, 4))
        
#        Symetry Constraint || self.conv1_f3 = self.conv1_f1 || self.conv1_f4 = self.conv1_f0
#        self.conv1_f3 = nn.Conv2d(1,  64, (9, 9), (1, 1), (0, 0))
#        self.conv1_f4 = nn.Conv2d(1,  64, (9, 9), (1, 1), (0, 0))
        
        self.conv2 = nn.Conv2d(320, 32, (5, 5), (1, 1), (2, 2))
        self.conv3 = nn.Conv2d(32, 1,  (5, 5), (1, 1), (2, 2))
        
        self.srcnn_model = srcnn_model
        self.upscale_factor = upscale_factor
        
        self._initialize_weights()

    def forward(self, x):
        
        h10 = x[:,[0],:,:]
        h11 = x[:,[1],:,:]
        h12 = x[:,[2],:,:]
        h13 = x[:,[3],:,:]
        h14 = x[:,[4],:,:]        
        
#        h10 = h10.view(h10.size[0], 1, h10.size[1], h10.size[2])
#        h11 = h11.view(h11.size[0], 1, h10.size[1], h10.size[2])
#        h12 = h12.view(h12.size[0], 1, h10.size[1], h10.size[2])
#        h13 = h13.view(h13.size[0], 1, h10.size[1], h10.size[2])
#        h14 = h14.view(h14.size[0], 1, h10.size[1], h10.size[2])
        
        h10 = self.conv1_f0(h10)
        h11 = self.conv1_f1(h11)
        h12 = self.conv1_f2(h12)
        h13 = self.conv1_f1(h13)
        h14 = self.conv1_f0(h14) 
     
        x = F.relu(torch.cat((h10, h11, h12, h13, h14), 1))
        
        x = F.relu(self.conv2(x))
        x = (self.conv3(x))
        return x
    
    def _initialize_weights(self):
        
        srcnn_model = torch.load(self.srcnn_model, map_location=lambda storage, loc: storage) # forcing to load to CPU
        
        
        
        self.conv1_f0.weight.data = (srcnn_model.conv1.weight.data).clone()
        self.conv1_f1.weight.data = (srcnn_model.conv1.weight.data).clone()
        self.conv1_f2.weight.data = (srcnn_model.conv1.weight.data).clone()
        
        self.conv1_f0.bias.data = (srcnn_model.conv1.bias.data).clone()
        self.conv1_f1.bias.data = (srcnn_model.conv1.bias.data).clone()
        self.conv1_f2.bias.data = (srcnn_model.conv1.bias.data).clone()
        
        self.conv2.weight.data = torch.cat((srcnn_model.conv2.weight.data, 
                                            srcnn_model.conv2.weight.data, 
                                            srcnn_model.conv2.weight.data, 
                                            srcnn_model.conv2.weight.data, 
                                            srcnn_model.conv2.weight.data), 1).clone()/5.0
        self.conv2.bias.data = (srcnn_model.conv2.bias.data).clone()
        

        self.conv3.weight.data = (srcnn_model.conv3.weight.data).clone()
        self.conv3.bias.data = (srcnn_model.conv3.bias.data).clone()
        


if __name__ == "__main__":
    model = Net(upscale_factor=4, srcnn_model='epochs_SRCNN/model_epoch_800.pth')
    print(model)
