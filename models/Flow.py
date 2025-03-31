import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

class Flow_enhance(nn.Module):
    def __init__(self):
        super(Flow_enhance, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=1)  # 输出通道数调整为3
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3,padding=1)  # 输出通道数调整为3
        self.sigmoid = nn.Sigmoid()  # 用于将注意力图映射到 [0, 1] 范围
        self.bn2 = nn.BatchNorm2d(3)  # 对conv2的输出进行BatchNorm
        self.relu = nn.ReLU()  # ReLU激活函数

    def forward(self, image, flow_image):
        # 在通道维度拼接
        F = torch.cat((image, flow_image), dim=1)  # 拼接在通道维度

        F1 = self.conv1(F) # [batch_size, 1, height, width]
        F1 = self.sigmoid(F1)
        F2 = self.conv2(F)  # [batch_size, 3, height, width]
        F2 = self.bn2(F2)  # BatchNorm层
        F2 = self.relu(F2)  # ReLU激活
        # 逐元素相乘得到注意力图
        attention_map = F1 * F2  # [batch_size, 3, height, width]

        # 加权图像
        x = attention_map  + image  # 使用注意力图来调整原始图像

        return x
def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean


def get_flow(img1,img2,Flow):
    div_flow = 20
    div_size = 64

    img1, img2, _ = centralize(img1, img2)

    height, width = img1.shape[-2:]
    orig_size = (int(height), int(width))

    if height % div_size != 0 or width % div_size != 0:
        input_size = (
            int(div_size * np.ceil(height / div_size)),
            int(div_size * np.ceil(width / div_size))
        )
        img1 = F.interpolate(img1, size=input_size, mode='bilinear', align_corners=False)
        img2 = F.interpolate(img2, size=input_size, mode='bilinear', align_corners=False)
    else:
        input_size = orig_size

    input_t = torch.cat([img1, img2], 1).cuda()

    output = Flow(input_t).data

    flow = div_flow * F.interpolate(output, size=input_size, mode='bilinear', align_corners=False) #(B,2,H,W)

    if input_size != orig_size:
        scale_h = orig_size[0] / input_size[0]
        scale_w = orig_size[1] / input_size[1]
        flow = F.interpolate(flow, size=orig_size, mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= scale_w
        flow[:, 1, :, :] *= scale_h

    # flow = flow[0].cpu().permute(1, 2, 0).numpy()

    return flow