import einops
import math
from data import build_dataset,build_sampler,build_dataloader
from GIP import build
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from models.criterion import build as build_id_criterion
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from GIP import GIP
from collections import deque
import time
from structures.instances import Instances
from models.utils import load_checkpoint
from PIL import Image
import cv2
from torchvision import transforms
import numpy as np
from models.utils import save_checkpoint,get_model
import os
from eval_engine import evaluate_one_epoch
from models.ReID import ReID
from models.Flow import get_flow,Flow_enhance
from torch.utils.tensorboard import SummaryWriter
from FastFlowNet.models.FastFlowNet import FastFlowNet
from FastFlowNet.flow_vis import flow_to_color
from models.feature_embedding import SimpleCNN

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.utils import labels_to_one_hot, is_distributed, distributed_rank, \
    combine_detr_outputs, detr_outputs_index_select, infos_to_detr_targets, batch_iterator, is_main_process
import torch.distributed

#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
#os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

import warnings
warnings.filterwarnings("ignore")

transform = transforms.Compose([
     # 随机裁剪，裁剪区域比例范围为80%到100%
    transforms.RandomAffine(
        degrees=0,  # 不进行旋转
        scale=(0.8, 1.2),  # 随机缩放
        translate=(0.1, 0.1)  # 随机平移，最多10%图片宽度和高度
    ), 
])
transform1 = transforms.Compose([
    transforms.Resize((384, 128)),  # 将图像缩放到 32x32
])

transform2 = transforms.Compose([
transforms.Resize((256, 128)),  
#transforms.CenterCrop(32), 
])

class TPS:
    """
    Time Per Step.
    """
    def __init__(self, windows_size: int = 50):
        self.tps_deque = deque(maxlen=windows_size)     # time per step.

    def update(self, tps: float):
        self.tps_deque.append(tps)

    @property
    def average(self):
        tps_list = list(self.tps_deque)
        return sum(tps_list) / len(tps_list)

    def eta(self, total_steps: int, current_steps: int):
        return self.average * (total_steps - current_steps)

    @classmethod
    def timestamp(cls):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

    @classmethod
    def format(cls, seconds: float):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h)}:{int(m)}:{int(s)}"

def train(config: dict):

    dataset_train = build_dataset(config=config)
    model = build(config=config)
    model.to(device=torch.device(config["DEVICE"]))
    

    # For optimizer:
    param_groups = get_param_groups(model=model, config=config)
    optimizer = AdamW(params=param_groups, lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])

    # Criterion (Loss Function):
    id_criterion = build_id_criterion(config=config)

    # Scheduler:
    if config["SCHEDULER_TYPE"] == "MultiStep":
        scheduler = MultiStepLR(optimizer, milestones=config["SCHEDULER_MILESTONES"],
                                gamma=config["SCHEDULER_GAMMA"])
    else:
        raise RuntimeError(f"Do not support scheduler type {config['SCHEDULER_TYPE']}.")

    # Train States:
    train_states = {
        "start_epoch": 0,
        "global_iter": 0
    }
    
    if config['PERTRAIN']:
        if not config.get('PERTRAIN_PATH'):
            raise ValueError("The 'PERTRAIN_PATH' in the config is empty or not set. Please provide a valid checkpoint path.")
        load_checkpoint(model,config['PERTRAIN_PATH'],train_states,optimizer,scheduler)
    CNN = model.CNN
    CNN.to(device=torch.device(config["DEVICE"]))
    FLOW = FastFlowNet().eval()
    FLOW.to(device=torch.device(config["DEVICE"]))
    #enhance = model.enhance
    #enhance.to(device=torch.device(config["DEVICE"]))
    FLOW.load_state_dict(torch.load('./FastFlowNet/checkpoints/fastflownet_ft_mix.pth',map_location='cuda:{}'.format(distributed_rank())))
    p_embedding = model.p_embedding.to(device=torch.device(config["DEVICE"]))
    # For resume:
    if train_states["start_epoch"] > 0:
        for i in range(0, train_states["start_epoch"]):
            scheduler.step()

  # Distributed, every gpu will share the same parameters.
    if is_distributed():
        model = DDP(model, device_ids=[distributed_rank()])
  
        
    for epoch in range(train_states["start_epoch"], config["EPOCHS"]):

        epoch_start_timestamp = TPS.timestamp()
        dataset_train.set_epoch(epoch)
        sampler_train = build_sampler(dataset=dataset_train, shuffle=True)
        dataloader_train = build_dataloader(
           dataset=dataset_train,
           sampler=sampler_train,
           batch_size=config["BATCH_SIZE"],
           num_workers=config["NUM_WORKERS"]
       )
        if is_distributed():
            sampler_train.set_epoch(epoch)
        
        
        #Train one epoch:
        train_one_epoch(
        config=config, model=model,CNN_model=CNN,Flow = FLOW, #enhance = enhance,
        p_embedding=p_embedding,
        dataloader=dataloader_train, id_criterion=id_criterion,
        optimizer=optimizer, epoch=epoch, states=train_states,
        clip_max_norm=config["CLIP_MAX_NORM"], detr_num_train_frames=config["DETR_NUM_TRAIN_FRAMES"],
        detr_checkpoint_frames=config["DETR_CHECKPOINT_FRAMES"],
        lr_warmup_epochs=0 if "LR_WARMUP_EPOCHS" not in config else config["LR_WARMUP_EPOCHS"]
         )

        lr = optimizer.state_dict()["param_groups"][-1]["lr"]

        time_per_epoch = TPS.format(TPS.timestamp() - epoch_start_timestamp)
        print(f"[Epoch {epoch} Finish] [Total Time: {time_per_epoch}] ")

        # Save checkpoint.
        if (epoch + 1) % config["SAVE_CHECKPOINT_PER_EPOCH"] == 0:
            os.makedirs(os.path.dirname(config["OUTPUTS_DIR"]), exist_ok=True)
            save_checkpoint(model=model,
                           path=os.path.join(config["OUTPUTS_DIR"], f"checkpoint_{epoch}.pth"),
                           states=train_states,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           )
            if config["INFERENCE_DATASET"] is not None :
                continue
                evaluate_one_epoch(
            config=config,
            model=model,
            CNN=CNN,
            Flow=FLOW, p_embedding=p_embedding,
            dataset=config["INFERENCE_DATASET"],
            data_split=config["INFERENCE_SPLIT"],
            outputs_dir=eval_outputs_dir,
            # dataloader=dataloader_train,
        )




        # Next step.
        scheduler.step()


def train_one_epoch(config: dict, model: GIP, CNN_model :SimpleCNN, Flow :FastFlowNet,p_embedding:nn.Module, #enhance : Flow_enhance,
                    dataloader: DataLoader, id_criterion: nn.Module,
                    optimizer: torch.optim,
                    epoch: int, states: dict, clip_max_norm: float, detr_num_train_frames: int,
                    detr_checkpoint_frames: int = 0, lr_warmup_epochs: int = 0):
    model.train()
    writer = SummaryWriter(log_dir=os.path.join(config["OUTPUTS_DIR"], f"result"))
    tps = TPS() # save time per step
    CNN = CNN_model.eval()
    device = torch.device(config["DEVICE"])
    p_embedding = p_embedding

    optimizer.zero_grad()  # init optim
    iter_start_timestamp = TPS.timestamp()
    losses = []

    for i, batch in enumerate(dataloader):

        if epoch < lr_warmup_epochs:
            # Do lr warmup:
            lr_warmup(optimizer=optimizer, epoch=epoch, iteration=i,
                      orig_lr=config["LR"], warmup_epochs=lr_warmup_epochs, iter_per_epoch=len(dataloader))


      # prepare some meta info
        num_gts = sum([len(info["boxes"]) for info in batch["infos"][0]])

        B, T = len(batch["images"]), len(batch["images"][0])
        detr_num_train_frames = min(detr_num_train_frames, T)
        frames = batch["nested_tensors"]  # (B, T, C, H, W) for tensors
        infos = batch["infos"]
        assert B==1, "f:for simple B==1"
        random_frame_idxs = torch.randperm(T)
        detr_train_frame_idxs = random_frame_idxs[:detr_num_train_frames]
        detr_no_grad_frame_idxs = random_frame_idxs[detr_num_train_frames:]

        all_ids = [_["ids"] for _ in infos[0]]
        all_ids_in_one_list = torch.cat(all_ids, dim=0).tolist()
        all_ids_set = set(all_ids_in_one_list)
        N = len(all_ids_set)
        all_boxes = [_["boxes"].detach().to(device) for _ in infos[0]]
        all_images = [_.detach().to(device) for _ in batch["images"][0]]
        #all_images = [_["unnorm_image_tensor"].detach().to(device) for _ in infos[0]]
        box_dim = all_boxes[0].shape[-1]
        # 制作allbox，allmask，裁剪完之后全部送CNN再mask
        # Build a mapping from ID to index, and index to ID:
        id_to_idx = {list(all_ids_set)[_]: list(range(N))[_] for _ in range(N)}
        idx_to_id = {v: k for k, v in id_to_idx.items()}
        all_mask = []
        all_crop = []
        all_flow = []
        all_box = []
        images = torch.stack(all_images)
        images_first = images[:-1]
        images_first = torch.cat((images[:1],images_first),dim=0)
        images_second= images
        with torch.no_grad():
            FLOW_images = get_flow(images_first,images_second,Flow.eval()).detach()

            predicted_flow = -FLOW_images.permute(0, 2, 3, 1)
            bz, h, w, c = predicted_flow.shape
            

        FLOW_images[:, 0, :, :] = FLOW_images[:, 0, :, :] / w #归一化
        FLOW_images[:, 1, :, :] = FLOW_images[:, 1, :, :] / h
        for t in range(T):
            image = images[t]
            #if t == 0:
            #    image = torch.cat((image,image),dim=0)
            #else:
            #    image = torch.cat((images[t-1],image),dim=0)
            Flow_image = FLOW_images[t]
            height, width = image.shape[-2:]
            scale = torch.as_tensor([width, height, width, height], device=device)
            t_idxs = torch.tensor(
                [id_to_idx[_id.item()] for _id in all_ids[t]], dtype=torch.long, device=device
            )  # which index to use, for each object in current frame "t"
            t_token_mask = torch.ones((N,), dtype=torch.bool, device=device)
            t_token_mask[t_idxs] = False
            all_mask.append(t_token_mask)
            t_boxes = torch.zeros((N, box_dim), dtype=torch.float, device=device)
            boxes = all_boxes[t]
            crop_frame = []
            flow_frame = []
      
            for box in boxes:
                x, y, w, h = box * scale
                
                
                # 计算裁剪框的索引
                y_min = max(0, math.floor(y - h / 2))
                y_max = min(image.shape[1], math.ceil(y + h / 2))
                x_min = max(0, math.floor(x - w / 2))
                x_max = min(image.shape[2], math.ceil(x + w / 2))
                if y_min >= image.shape[1]:
                    y_min = y_max - 10
                if y_max <= 0:
                    y_max = y_min + 10
                if x_min >= image.shape[2]:
                    x_min = x_max - 10
                if x_max <= 0:
                    x_max = x_min + 10
                
               
                
                # 执行裁剪
                image_Crop = image[:, y_min:y_max, x_min:x_max]
                flow_Crop = Flow_image[:, y_min:y_max, x_min:x_max]
          
                flow_Crop = transform2(flow_Crop)
                
                def lvy():
                    # 定义用于逆标准化的均值和方差
                    mean = np.array([0.485, 0.456, 0.406])  # 根据你的标准化方法调整
                    std = np.array([0.229, 0.224, 0.225])   # 根据你的标准化方法调整
                    
                    # 逆标准化操作
                    image_np = image_Crop.cpu().permute(1, 2, 0).numpy()
                    
                    # 逆变换：乘以标准差并加上均值
                    image_np = image_np * std + mean
                    # 保存图像
                    output_folder = "saved_images"
                    os.makedirs(output_folder, exist_ok=True)  # 如果文件夹不存在，则创建
                    
                    # 转换为 PIL 图像格式
                    image_pil = Image.fromarray(np.uint8(image_np*255))
                    
                    # 设置保存的路径
                    output_path = os.path.join(output_folder, str(box)+"flow_crop_image.png")
                    
                    # 保存图像
                    image_pil.save(output_path)
                # 2. 显示图像
                # plt.imshow(image_np)
                # plt.axis('off')  # 关闭坐标轴
                # plt.show()
                #mv = torch.cat((flow_Crop.flatten(), (x/image.shape[2]).unsqueeze(0),(y/image.shape[1]).unsqueeze(0)))
                
                mv = flow_Crop
                flow_frame.append(mv)
                image_Resize = transform1(image_Crop)
                crop_frame.append(image_Resize)

            flow_frame = torch.stack(flow_frame, dim=0)
            crop_frame = torch.stack(crop_frame, dim=0)
            size = crop_frame[0].shape
            t_box = torch.zeros((N, 4), dtype=torch.float, device=device)
            t_box[t_idxs] = boxes
            t_crop = torch.zeros((N, *size), dtype=torch.float, device=device)
            t_crop[t_idxs] = crop_frame
            size1 = flow_frame[0].shape
            t_flow = torch.zeros((N, *size1), dtype=torch.float, device=device)
            t_flow[t_idxs] = flow_frame
            all_crop.append(t_crop)
            all_flow.append(t_flow)
            all_box.append(t_box)
        
            
        all_flow = torch.stack(all_flow, dim=0).view(-1, 2, 256, 128)
        all_box = torch.stack(all_box, dim=0)
        all_mask = torch.stack(all_mask, dim=0)
        all_crop = torch.stack(all_crop, dim=0).view(-1, 3, 384, 128)
       
        output1 = all_crop
        #output1, output2 = torch.split(all_crop, 3, dim=1)
        with torch.no_grad():
            output1 = CNN(output1).detach()
        #    output2 = CNN(output2)
        # output = torch.cat((output, all_flow), dim=-1)
        
        #all_flow = all_flow.view(output2.shape[0],-1)
        #all_flow = all_flow.unsqueeze(2).unsqueeze(3)
        
        output1 = output1.unsqueeze(2).unsqueeze(3)
        #output2 = output2.unsqueeze(2).unsqueeze(3)
          
        output = get_model(model).GFFfuse(output1,all_flow).view(T,N,-1)
        position_embedding = p_embedding(all_box)
        #print(position_embedding.shape)
        output = output + position_embedding
        
        
        CNN_output1 = []
        for t in range(T):
            CNN_output1.append(output[t][~all_mask[t]])
            continue
            if t in detr_no_grad_frame_idxs:
                with torch.no_grad():
                    CNN_output1.append(output[t][~all_mask[t]])
            if t in detr_train_frame_idxs:
                CNN_output1.append(output[t][~all_mask[t]])
        
        
        

        CNN_outputs={"pred_boxes":all_boxes,"feature":CNN_output1}

        match_instances = generate_match_instances(
                     infos=infos, CNN_outputs=CNN_outputs
                )

        


        assert len(match_instances) == 1, f"For simplicity, only the case of bs=1 is implemented."
        get_model(model).add_random_id_words_to_instances(instances=match_instances[0])
        pred_id_words, gt_id_words, id_gts, ap_feature, mask, history= get_model(model).forward_train(
            track_history=match_instances,
            traj_drop_ratio=config["TRAJ_DROP_RATIO"],
            traj_switch_ratio=config["TRAJ_SWITCH_RATIO"] if "TRAJ_SWITCH_RATIO" in config else 0.0,
            use_checkpoint=config["SEQ_DECODER_CHECKPOINT"],
        )

        # Calculate the overall loss for barkward processing:
        id_loss = id_criterion(pred_id_words, gt_id_words)
        # new_id = config['NUM_ID_VOCABULARY']
        # ReID_loss = ReID_losses(ap_embed, id_gts, mask, history,new_id)
        useloss = config['USE_REID']
        ap_embed = ap_feature
        
        if useloss == True:
            reid_loss = ReID(config,history).to(device)
            id_guide_matrix = reid_loss(ap_embed)
            # 创建一个全是 2 的 tensor
            twos_tensor = torch.full((id_guide_matrix.shape[0],1,id_guide_matrix.shape[2]), 2).to(device)
            id_guide_matrix=torch.cat((id_guide_matrix,twos_tensor),dim=1)
            pred_id = pred_id_words.squeeze()
            id_guide_matrix = id_guide_matrix.permute(0, 2, 1)[~mask]
            # 对pred_id进行切片和softmax
            pred_onehot = gt_id_words[:, :, :].squeeze()
            pred_id = F.softmax(pred_id, dim=1)
          
            if config['USE_LOSS2'] == True:
               pred_id = pred_id.clone()  # 在操作之前创建一个副本，避免in-place修改
               pred_id[pred_onehot.bool()] = -pred_id[pred_onehot.bool()]
               pred_id = -pred_id*config["ID_GUIDE_REID"] + pred_onehot*config["REID_LOSS_WEIGHT"] #这个和下面选一个
            else:
              pred_id = pred_id*config["ID_GUIDE_REID"] + pred_onehot*config["REID_LOSS_WEIGHT"]
            
            pred_loss = torch.sum(pred_id * id_guide_matrix, dim=1, keepdim=True).mean()
            loss = id_loss * id_criterion.weight   + pred_loss
            
            writer.add_scalar("id_loss", id_loss.detach(), states["global_iter"])
            writer.add_scalar("id_guide_reid", pred_loss.detach(), states["global_iter"])
            writer.add_scalar("loss", loss.detach(), i)

    
            
        else:
            loss = id_loss * id_criterion.weight
          
            writer.add_scalar("id_loss", id_loss.detach(), states["global_iter"])
            writer.add_scalar("loss", loss.detach(), states["global_iter"])
            
        losses.append(id_loss.item())


        # Backward the loss:
        loss /= config["ACCUMULATE_STEPS"]
        loss.backward()

        # Parameters update:
        if (i + 1) % config["ACCUMULATE_STEPS"] == 0:
            optimizer.step()
            optimizer.zero_grad()

        iter_end_timestamp = TPS.timestamp()
        tps.update(iter_end_timestamp - iter_start_timestamp)
        eta = tps.eta(total_steps=len(dataloader), current_steps=i)

        
        if useloss == True:
            print(f"\r[Epoch: {epoch}] [{i}/{len(dataloader)}] [tps: {tps.average:.2f}s] [eta: {TPS.format(eta)}] [loss: {loss:.4f}] [id_loss: {id_loss:.4f}] [ID_Guide_ReID: {pred_loss:.4f}]", end="")
        else:
            print(f"\r[Epoch: {epoch}] [{i}/{len(dataloader)}] [tps: {tps.average:.2f}s] [eta: {TPS.format(eta)}] [loss: {loss:.4f}]", end="")
        
        
        iter_start_timestamp = TPS.timestamp()
        states["global_iter"] += 1
        
    avg_loss = sum(losses) / len(losses)
    print()
    print("avg_id_loss:" , avg_loss)
    writer.close()
    states["start_epoch"] += 1


def generate_match_instances( infos, CNN_outputs):
    match_instances = []
    B, T = len(infos), len(infos[0])
    for b in range(B):
        match_instances.append([])
        for t in range(T):
            flat_idx = b * T + t
            instances = Instances(image_size=(0, 0))
            instances.ids = infos[b][t]["ids"]
            instances.gt_boxes = infos[b][t]["boxes"]
            instances.pred_boxes = CNN_outputs["pred_boxes"][flat_idx]
            instances.outputs = CNN_outputs["feature"][flat_idx]
            match_instances[b].append(instances)
    return match_instances



def get_param_groups(model: nn.Module, config) -> list[dict]:
    def match_names(name, key_names):
        for key in key_names:
            if key in name:
                return True
        return False
    # keywords
    backbone_names = config["LR_BACKBONE_NAMES"]
    linear_proj_names = config["LR_LINEAR_PROJ_NAMES"]
    dictionary_names = [] if "LR_DICTIONARY_NAMES" not in config else config["LR_DICTIONARY_NAMES"]
    _dictionary_scale = 1.0 if "LR_DICTIONARY_SCALE" not in config else config["LR_DICTIONARY_SCALE"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if match_names(n, backbone_names) and p.requires_grad],
            "lr_scale": config["LR_BACKBONE_SCALE"],
            "lr": config["LR"] * config["LR_BACKBONE_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if match_names(n, linear_proj_names) and p.requires_grad],
            "lr_scale": config["LR_LINEAR_PROJ_SCALE"],
            "lr": config["LR"] * config["LR_LINEAR_PROJ_SCALE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if match_names(n, dictionary_names) and p.requires_grad],
            "lr_scale": _dictionary_scale,
            "lr": config["LR"] * _dictionary_scale
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if not match_names(n, backbone_names)
                       and not match_names(n, linear_proj_names)
                       and not match_names(n, dictionary_names)
                       and p.requires_grad],
        }
    ]
    return param_groups

def lr_warmup(optimizer, epoch: int, iteration: int, orig_lr: float, warmup_epochs: int, iter_per_epoch: int):
    # min_lr = 1e-8
    total_warmup_iters = warmup_epochs * iter_per_epoch
    current_lr_ratio = (epoch * iter_per_epoch + iteration + 1) / total_warmup_iters
    current_lr = orig_lr * current_lr_ratio
    for param_grop in optimizer.param_groups:
        if "lr_scale" in param_grop:
            param_grop["lr"] = current_lr * param_grop["lr_scale"]
        else:
            param_grop["lr"] = current_lr
        pass
    return

