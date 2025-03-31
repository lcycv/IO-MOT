import torch
import torch.nn as nn
import torch.nn.functional as F

class FuseGFFConvBlock(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_size=(1, 1), stride=1):
        super(FuseGFFConvBlock, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels, n_filters, kernel_size=kernel_size, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(n_filters)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        # Apply first convolution, batch normalization, and ReLU activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Apply second convolution, batch normalization, and ReLU activation
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x
        

         
class FlowGFFConvBlock(nn.Module):
    def __init__(self):
        super(FlowGFFConvBlock, self).__init__()
        # 初始卷积层，用于从输入光流图像中提取初步特征
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3)  # 输出 (64, 128, 64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 输出 (128, 64, 32)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # 一个更深的卷积层
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 输出 (512, 16, 8)
        # 最后使用全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 输出尺寸为 (512, 1, 1)
        # 全连接层输出 256 维特征
        self.fc = FuseGFFConvBlock(512, 256)
        #self.fc =  nn.Conv2d(512, 256, kernel_size=1)
    
      
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.global_pool(x)

        x = self.fc(x)
       
        return x

class GFF(nn.Module):
    def __init__(self):
        super(GFF, self).__init__()
        # Define the ResNetBlock_1 (this would be specific to your model)
        # Define the 1x1 convolutions
        self.conv1x1_1 = nn.Conv2d(2048, 256, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(2048, 256, kernel_size=1)
        self.conv1x1_4 = nn.Conv2d(256, 256, kernel_size=1)
        self.f1 = FuseGFFConvBlock(256, 256)
        
        self.FlowGFF = FlowGFFConvBlock()
        self.conv1x1_5 = nn.Conv2d(256, 256, kernel_size=1)
        self.f3 =  FuseGFFConvBlock(512, 256)
        
    def forward(self, reid1, flow):
        # x1 is the input tensor (reid)
        x1 = reid1
        # Apply 1x1 convolutions without activation or normalization
        x1n = self.conv1x1_1(x1)
        g1 = self.conv1x1_2(x1n)
        # Apply sigmoid activation function
        g1 = torch.sigmoid(g1)
        
       
        
        
        x3 = flow
        x3n = self.FlowGFF(x3)
        g3 = self.conv1x1_5(x3n)
        g3 = torch.sigmoid(g3)
    
        # Fuse the features from reid and flow
        x1gff = (1 + g1) * x1n  + (1 - g1) * (g3 * x3n)
       
        x3gff = (1 + g3) * x3n + (1 - g3) * (g1 * x1n)
        
        # Apply FuseGFFConvBlock
        x1gff = self.f1(x1gff)  # Fix the input channels and apply the block
       # Fix the input channels and apply the block
        # Concatenate the results along the channel dimension (dim=1)

        output = torch.cat([x1gff,  x3gff], dim=1)
        output = self.f3(output)
        
        
        
        
        
        return output

