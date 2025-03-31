import torch.nn as nn

from collections import OrderedDict
from pathlib import Path
import os
import torchvision.transforms as T
import torch
import cv2
import torchvision

import numpy as np

from external.adaptors.fastreid_adaptor import FastReID

class SimpleCNN(nn.Module):
    def __init__(self, dataset = 'dance'):
        super(SimpleCNN, self).__init__()
        self.dataset = dataset

        if self.dataset == "mot":
            #if self.test_dataset:
            path = "external/weights/mot17_sbs_S50.pth"
           # else:
           #     return self._get_general_model()
        elif self.dataset == "mot20":
            if self.test_dataset:
                path = "../external/weights/mot20_sbs_S50.pth"
            else:
                return self._get_general_model()
        elif self.dataset == "dance":
            
            path = "external/weights/dance_sbs_S50.pth"
            #path = "/home1/lcy/FIP-v3/external/weights/model_final.pth"
            # path = "/home/estar/lwy/DiffMOT/external/weights/dancetrack_sbs_S50_hybtid.pth"
        elif self.dataset == "sports":
            path = "external/weights/sports_sbs_S50.pth"
        else:
            raise RuntimeError("Need the path for a new ReID model.")

        model = FastReID(path)
        model.cuda()
        model.half()
        self.model = model.eval()
        


    def forward(self, x):
        with torch.no_grad():
            # 如果是 mot17 数据集，先调整输入图像的大小
            if self.dataset == "mot17":
                resize = T.Resize((256, 128))  # 目标大小为 256x128
                x = resize(x)  # 调整图像大小
        
            self.model.eval()
            x = self.model(x).float()
            x = nn.functional.normalize(x, dim=-1)
        return x


       

    


    def _get_general_model(self):
        """Used for the half-val for MOT17/20.

        The MOT17/20 SBS models are trained over the half-val we
        evaluate on as well. Instead we use a different model for
        validation.
        """
        model = torchreid.models.build_model(name="osnet_ain_x1_0", num_classes=2510, loss="softmax", pretrained=False)
        sd = torch.load("../external/weights/osnet_ain_ms_d_c.pth.tar")["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in sd.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        model.eval()
        model.cuda()
        self.model = model
        self.crop_size = (128, 256)
        self.normalize = True


# 示例使用
if __name__ == "__main__":
    model = SimpleCNN().cuda()  # 假设有 10 个类别
    sample_input = torch.randn(10, 3, 256, 128).cuda()  # 输入一个 32x32 的 RGB 图像
    output = model(sample_input)
    print(output.shape)  # 应该是 [1, 10]