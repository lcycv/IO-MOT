import torch
import einops
import torch.nn as nn
import numpy as np

from utils.utils import is_distributed, distributed_world_size
import torch.nn.functional as F

class ReID(nn.Module):
    def __init__(self,config, history: dict):
        super().__init__()
        self.history = history
        self.num = config['NUM_ID_VOCABULARY']
        self.device =torch.device(config["DEVICE"])
        masks = self.history["masks"]
        self.masks = masks
        times = self.history["times"]
        self.ori_time = self.history["times"]
        ids = self.history["ids"]
        features = self.history["features"]
        self.times = times

        non_minus_one_elements = []
        mask = []
        for row in ids:
            non_minus_one = row[row != -1]
            if non_minus_one.numel() == 0:
                non_minus_one_elements.append(999)
                mask.append(False)
            else:
                non_minus_one_elements.append(non_minus_one[0].item())
                mask.append(True)
        mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
        self.mask_now = mask
        self.ids = torch.tensor(non_minus_one_elements, device=self.device)
        self.feature = features



    def forward(self, ap_embed):
        # now_time = self.ori_time[~mask]
        B,T = self.ori_time.shape


        cosine = torch.full((B,self.num, T), 2).float().to(self.device)
        decoder_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=T, device=ap_embed.device
        )
        decoder_mask = ~(torch.exp(decoder_mask).to(torch.bool))
        decoder_mask = decoder_mask[:, None, :].repeat((1, B, 1))
        masks =self.masks[None,:,:].repeat((B, 1,  1))
        for t in range(T):
            his_mask = decoder_mask[t].repeat(B,1,1)
            ap_now = ap_embed[:,t,:]
            ap_now = ap_now.unsqueeze(1)
            #cosine_sim = 2- F.cosine_similarity(ap_now[:,:,:,:,:128].unsqueeze(1), self.feature[:,:,:,:,:128].unsqueeze(0), dim=-1)-F.cosine_similarity(ap_now[:,:,:,:,128:].unsqueeze(1), self.feature[:,:,:,:,128:].unsqueeze(0), dim=-1)  #(history,T,now)
            cosine_sim = 1- F.cosine_similarity(ap_now.unsqueeze(1), self.feature.unsqueeze(0), dim=-1)
            cosine_sim[his_mask | masks] = 2

            min_values = cosine_sim.min(dim=-1).values
            id_now = self.ids
            cosine[:,id_now[self.mask_now],t]=min_values[:,self.mask_now]

        return cosine
        
    def forward1(self, ap_embed):
        B, T = self.ori_time.shape


        cosine = torch.full((B, T, self.num), 2).float().to(self.device)


        decoder_mask = nn.Transformer.generate_square_subsequent_mask(sz=T, device=ap_embed.device)
        decoder_mask = ~(torch.exp(decoder_mask).to(torch.bool))
        decoder_mask = decoder_mask.unsqueeze(0).repeat(B, 1, 1)  # (B, T, T)
        decoder_mask = decoder_mask[:,:,None,:].repeat(1,1,B,1)

        masks = self.masks[None,None, :, :].repeat(B, T, 1, 1)  # (B, T, T)

        ap_embed = ap_embed.unsqueeze(2).unsqueeze(3)  # (B, T, 1, D)
        feature_expanded = self.feature.unsqueeze(0).unsqueeze(0)  # (1, 1, num_features)

        cosine_sim = 1 - F.cosine_similarity(ap_embed, feature_expanded, dim=-1)

        cosine_sim[decoder_mask | masks] = 2

        min_values = cosine_sim.min(dim=-1).values  # (B, T)

        id_now = self.ids
        cosine[:, :, id_now[self.mask_now]] = min_values[:, :,self.mask_now]

        return cosine.permute(0,2,1)

