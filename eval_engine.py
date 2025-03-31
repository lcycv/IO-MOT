import torch.nn as nn
import torch
import os,cv2
from data.seq_dataset import SeqDataset
from torch.utils.data import DataLoader
from structures.ordered_set import OrderedSet
from collections import deque
from utils.nested_tensor import tensor_list_to_nested_tensor
from utils.box_ops import box_cxcywh_to_xyxy
from structures.instances import Instances
import math
from torchvision import transforms
# from models.Flow import Flow_enhance
from FastFlowNet.models.FastFlowNet import FastFlowNet
from models.Flow import get_flow
from models.feature_embedding import SimpleCNN
from GIP import build
import matplotlib.pyplot as plt
import numpy as np
import warnings
import time
from PIL import Image
import util

warnings.filterwarnings("ignore")

transform1 = transforms.Compose([
      transforms.Resize((256, 128)),
    #  transforms.CenterCrop(32),#, interpolation=Image.NEAREST),  # 使用较小内存消耗的插值方法
])
transform = transforms.Compose([
    # # 随机裁剪，裁剪区域比例范围为80%到100%
    # transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    # 缩放到32x32
    transforms.Resize((384, 128)),
     
])
from models.utils import load_checkpoint,get_model

@torch.no_grad()
def eval(config):
    model = build(config=config)
    model.to(device=torch.device(config["DEVICE"]))
    load_checkpoint(model, path=config["INFERENCE_MODEL"])
    CNN = model.CNN
    CNN.to(device=torch.device(config["DEVICE"]))
    FLOW = FastFlowNet().eval()
    FLOW.to(device=torch.device(config["DEVICE"]))
    
    p_embedding = model.p_embedding.to(device=torch.device(config["DEVICE"]))
    FLOW.load_state_dict(torch.load('./FastFlowNet/checkpoints/fastflownet_ft_mix.pth'))

    if config["INFERENCE_GROUP"] is not None:
        eval_outputs_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"], config["INFERENCE_GROUP"],
                                        config["INFERENCE_SPLIT"],
                                        f'{config["INFERENCE_MODEL"].split("/")[-1][:-4]}')
    else:
        eval_outputs_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"], "default", config["INFERENCE_SPLIT"],
                                        f'{config["INFERENCE_MODEL"].split("/")[-1][:-4]}')
    if config["INFERENCE_DATASET"] is not None:
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


@torch.no_grad()
def evaluate_one_epoch(config: dict, model: nn.Module, CNN: nn.Module,Flow :FastFlowNet, p_embedding:nn.Module,
                        dataset: str, data_split: str,
                       outputs_dir: str):
    model.eval()
    CNN.eval()
    Flow.eval()
    p_embedding.eval()
    device = config["DEVICE"]

    all_seq_names = get_seq_names(data_root=config["DATA_ROOT"], dataset=dataset, data_split=data_split)
    seq_names = [all_seq_names[_] for _ in range(len(all_seq_names))]
    flow_size = config['FLOW_SIZE']
    if len(seq_names) > 0:
        for i,seq in enumerate(seq_names):
            st = time.time()
            submit_one_seq(
                model=model,  CNN=CNN,dataset=dataset, Flow=Flow, p_embedding=p_embedding,
                flow_size = flow_size,
                seq_dir=os.path.join(config["DATA_ROOT"], dataset, data_split, seq),
                max_temporal_length=config["MAX_TEMPORAL_LENGTH"],
                outputs_dir=outputs_dir,
                det_thresh=config["DET_THRESH"],
                newborn_thresh=config["DET_THRESH"] if "NEWBORN_THRESH" not in config else config["NEWBORN_THRESH"],
                area_thresh=config["AREA_THRESH"], id_thresh=config["ID_THRESH"],
                image_max_size=config["INFERENCE_MAX_SIZE"] if "INFERENCE_MAX_SIZE" in config else 1333,
                inference_ensemble=config["INFERENCE_ENSEMBLE"] if "INFERENCE_ENSEMBLE" in config else 0,
                nms_max_overlap=config["NMS_MAX"] if "NMS_MAX" in config else 0.95,
            )
            ed = time.time()
            elapsed_time = ed - st
            print(f'[{i+1}/{len(seq_names)}],时间: {elapsed_time:.2f} s')

    else:
        submit_one_seq(
            model=model, CNN=CNN ,dataset=dataset, Flow=Flow, p_embedding=p_embedding,
            flow_size = flow_size,
            seq_dir=os.path.join(config["DATA_ROOT"], dataset, data_split, all_seq_names[0]),
            max_temporal_length=config["MAX_TEMPORAL_LENGTH"],
            outputs_dir=outputs_dir,
            det_thresh=config["DET_THRESH"],
            newborn_thresh=config["DET_THRESH"] if "NEWBORN_THRESH" not in config else config["NEWBORN_THRESH"],
            area_thresh=config["AREA_THRESH"], id_thresh=config["ID_THRESH"],
            image_max_size=config["INFERENCE_MAX_SIZE"] if "INFERENCE_MAX_SIZE" in config else 1333,
            fake_submit=True,
            inference_ensemble=config["INFERENCE_ENSEMBLE"] if "INFERENCE_ENSEMBLE" in config else 0,
        )

    tracker_dir =  os.path.join(outputs_dir, "tracker")
    dataset_dir = os.path.join(config["DATA_ROOT"], dataset)
    if dataset in ["DanceTrack", "SportsMOT"]:
        gt_dir = os.path.join(dataset_dir, data_split)
    elif dataset in ["MOT17_SPLIT", "MOT15", "MOT15_V2", "MOT17"]:
        gt_dir = os.path.join(dataset_dir, data_split)
    else:
        raise NotImplementedError(f"Do not support to find the gt_dir for dataset '{dataset}'.")

    print(data_split, "\n", os.path.join(dataset_dir, f'{data_split}_seqmap.txt', "\n", tracker_dir))
    # Need to eval the submit tracker:
    if dataset == "DanceTrack" or dataset == "SportsMOT":
        os.system(f"python TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_split}  "
                  f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
                  f"--SEQMAP_FILE {os.path.join(dataset_dir, f'{data_split}_seqmap.txt')} "
                  f"--SKIP_SPLIT_FOL True   --USE_PARALLEL True "
                  f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                  f"--TRACKERS_FOLDER {outputs_dir}")
    elif dataset == "MOT17" and data_split == "test":
        os.system(f"python TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_split}  "
                  f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
                  f"--SEQMAP_FILE {os.path.join(dataset_dir, f'{data_split}_seqmap.txt')} "
                  f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                  f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                  f"--TRACKERS_FOLDER {outputs_dir}")
    elif dataset == "MOT17_SPLIT" or dataset == "MOT17":
        print(f"python TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_split}  --METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} --SEQMAP_FILE {os.path.join(dataset_dir, f'{data_split}_seqmap.txt')} --SKIP_SPLIT_FOL True   --USE_PARALLEL True --NUM_PARALLEL_CORES 8 --PLOT_CURVES False --TRACKERS_FOLDER {outputs_dir} --BENCHMARK MOT17")
        os.system(f"python TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_split}  --METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} --SEQMAP_FILE {os.path.join(dataset_dir, f'{data_split}_seqmap.txt')} --SKIP_SPLIT_FOL True   --USE_PARALLEL True --NUM_PARALLEL_CORES 8 --PLOT_CURVES False --TRACKERS_FOLDER {outputs_dir} --BENCHMARK MOT17")
    elif dataset == "MOT15" or dataset == "MOT15_V2":
        os.system(f"python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_split}  "
                  f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
                  f"--SEQMAP_FILE {os.path.join(dataset_dir, f'{data_split}_seqmap.txt')} "
                  f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                  f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                  f"--TRACKERS_FOLDER {outputs_dir} --BENCHMARK MOT15")
    else:
        raise NotImplementedError(f"Do not support to eval the results for dataset '{dataset}'.")

    # Get eval Metrics:
    eval_metric_path = os.path.join(tracker_dir, "pedestrian_summary.txt")
    eval_metrics_dict = get_eval_metrics_dict(metric_path=eval_metric_path)

    return

@torch.no_grad()
def submit_one_seq(
            model: nn.Module,CNN:nn.Module, dataset: str, seq_dir: str, outputs_dir: str,Flow :FastFlowNet, p_embedding :nn.Module,
            flow_size : int,
            max_temporal_length: int = 0,
            det_thresh: float = 0.5, newborn_thresh: float = 0.5, area_thresh: float = 100, id_thresh: float = 0.1,
            image_max_size: int = 1333,
            fake_submit: bool = False,
            inference_ensemble: int = 0,
            nms_max_overlap :float =0.95
        ):
        
    os.makedirs(os.path.join(outputs_dir, "tracker"), exist_ok=True)
    seq_dataset = SeqDataset(seq_dir=seq_dir, dataset=dataset, width=image_max_size)
    seq_dataloader = DataLoader(seq_dataset, batch_size=1, num_workers=4, shuffle=False)
    # seq_name = seq_dir.split("/")[-1]
    seq_name = os.path.split(seq_dir)[-1]
    device = model.device
    current_id = 0
    ids_to_results = {}
    id_deque = OrderedSet()     # an ID deque for inference, the ID will be recycled if the dictionary is not enough.

    # Trajectory history:

    trajectory_history = deque(maxlen=max_temporal_length) #max_temporal_length)

    if fake_submit:
        print(f"[Fake] Start >> Submit seq {seq_name.split('/')[-1]}, {len(seq_dataloader)} frames ......",end="")
    else:
        print(f"Start >> Submit seq {seq_name.split('/')[-1]}, {len(seq_dataloader)} frames ......",end="")


    for i, (image, ori_image, prob, det ) in enumerate(seq_dataloader):

        ori_h, ori_w = image.shape[2], image.shape[3]
        scale = ori_image.shape[1] / ori_h
        frame = tensor_list_to_nested_tensor([image[0]]).to(device)
        #prob = prob.squeeze()
        #det = det.squeeze()
        prob=torch.tensor(prob)
        det=torch.tensor(det)
        det_idxs = [prob>det_thresh][0].flatten()
        boxes = det[det_idxs]
        prob = prob[det_idxs]
        b = np.array(boxes)
        b[:,2:]=b[:,2:]+b[:,:2]


        # indices = util.non_max_suppression(  # 非极大值抑制
        #     b, nms_max_overlap, np.array(prob.flatten()))
        # boxes = boxes[indices].to(device)
        # prob = prob[indices].to(device)
        boxes = boxes.to(device)
        prob = prob.to(device)

        crop_images = []
        flow_frame = []
        ori_image = image.to(device)
        if i == 0:
            images_first = ori_image
        images_second = ori_image
        record_images = ori_image
        FLOW_images = get_flow(images_first, images_second, Flow)

        predicted_flow = -FLOW_images.permute(0, 2, 3, 1)
        bz, h, w, c = predicted_flow.shape

        x_pos = torch.arange(w).reshape((1, w)).repeat((h, 1)).unsqueeze(0).to(device)
        y_pos = torch.arange(h).reshape((h, 1)).repeat((1, w)).unsqueeze(0).to(device)
  

        FLOW_images[:, 0, :, :] = FLOW_images[:, 0, :, :] / w  # 归一化
        FLOW_images[:, 1, :, :] = FLOW_images[:, 1, :, :] / h

        FLOW_images =  FLOW_images.squeeze()
        ori_image = ori_image.squeeze()
        #ori_image = torch.cat((images_first.squeeze(),ori_image),dim=0)
        all_boxes = []
        for box in boxes:
            
            x, y, w, h = box
        
            image_Crop = ori_image[
                         :,
                         max(0, math.floor(y )):min(ori_h, math.ceil(y + h )),
                         max(0, math.floor(x )):min(ori_w, math.ceil(x + w ))
                         ]
            # half_size = (int(flow_size) - 1) // 2

            image_np = image_Crop.cpu().permute(1, 2, 0).numpy()

            # 2. 显示图像
            # plt.imshow(image_np)
            # plt.axis('off')  # 关闭坐标轴
            # plt.show()
            
            

            flow_Crop = FLOW_images[
                        :,
                        max(0, math.floor(y)):min(ori_h, math.ceil(y + h)),
                        max(0, math.floor(x)):min(ori_w, math.ceil(x + w))
                        ]
                        
            all_boxes.append(torch.tensor([(x+w/2)/ori_w,(y+h/2)/ori_h,w/ori_w,h/ori_h]))
            flow_Crop = transform1(flow_Crop)
            # mv = torch.cat((flow_Crop.flatten(), ((x + w)/ori_w).unsqueeze(0), ((y + h)/ori_h).unsqueeze(0)))
            #mv = torch.cat((flow_Crop.flatten(), ((x + w/2) / ori_w).unsqueeze(0), ((y + h/2) / ori_h).unsqueeze(0)))
            mv = flow_Crop #.flatten()
            
            image_Resize = transform(image_Crop)
            crop_images.append(image_Resize)
            flow_frame.append(mv)
        if len(all_boxes) > 0:
            all_boxes = torch.stack(all_boxes, dim=0).float().to(device)
         
            flow_frame = torch.stack(flow_frame).float().to(device)
            crop_images = torch.stack(crop_images).float().to(device)
            output1 = crop_images
            #output1, output2 = torch.split(crop_images, 3, dim=1)
            output1 = CNN(output1)
            #output2 = CNN(output2)
            output1 = output1.unsqueeze(2).unsqueeze(3)
            #output2 = output2.unsqueeze(2).unsqueeze(3)
        
            flow_frame = flow_frame
            
            output = get_model(model).GFFfuse(output1,flow_frame).squeeze()
            
            position_embedding = p_embedding(all_boxes)
            
            
            output = output + position_embedding  
            
        else:
            output = torch.empty(0,256).to(device)
       
        box_results = boxes.float().to(device)
        box_results[:,2:] += box_results[:,:2]
        # 限制 x1, y1, x2, y2 的范围
        box_results = torch.stack([
            torch.clamp(box_results[:, 0], min=0, max=ori_w),  # x1 限制在 [0, ori_w] 之间
            torch.clamp(box_results[:, 1], min=0, max=ori_h),  # y1 限制在 [0, ori_h] 之间
            torch.clamp(box_results[:, 2], min=0, max=ori_w),  # x2 限制在 [0, ori_w] 之间
            torch.clamp(box_results[:, 3], min=0, max=ori_h)  # y2 限制在 [0, ori_h] 之间
        ], dim=1)
        # # De-normalize to target image size:
        # box_results = detr_det_boxes.cpu() * torch.tensor([ori_w, ori_h, ori_w, ori_h])
        # box_results = box_cxcywh_to_xyxy(boxes=box_results)
        #
        #
        #
        # Decoding the current objects' IDs



        assert max_temporal_length - 1 > 0, f"MOTIP need at least T=1 trajectory history, " \
                                            f"but get T={max_temporal_length - 1} history in Eval setting."
        current_tracks = Instances(image_size=(0, 0))
        current_tracks.boxes = box_results
        current_tracks.outputs = output
        current_tracks.ids = torch.tensor([model.num_id_vocabulary] * len(current_tracks),
                                          dtype=torch.long, device=current_tracks.outputs.device)
        current_tracks.confs = prob.to(device)
        trajectory_history.append(current_tracks)
        if len(trajectory_history) == 1:    # first frame, do not need decoding:
            newborn_filter = (trajectory_history[0].confs > newborn_thresh).reshape(-1, )   # filter by newborn
            trajectory_history[0] = trajectory_history[0][newborn_filter]
            box_results = box_results[newborn_filter.cpu()]
            ids = torch.tensor([current_id + _ for _ in range(len(trajectory_history[-1]))],
                               dtype=torch.long, device=current_tracks.outputs.device)
            trajectory_history[-1].ids = ids
            for _ in ids:
                ids_to_results[_.item()] = current_id
                current_id += 1
            id_results = []
            for _ in ids:
                id_results.append(ids_to_results[_.item()])
                id_deque.add(_.item())
            id_results = torch.tensor(id_results, dtype=torch.long)
        else:
            ids, trajectory_history, ids_to_results, current_id, id_deque, boxes_keep = model.inference(
                trajectory_history=trajectory_history,
                num_id_vocabulary=model.num_id_vocabulary,
                ids_to_results=ids_to_results,
                current_id=current_id,
                id_deque=id_deque,
                id_thresh=id_thresh,
                newborn_thresh=newborn_thresh,
                inference_ensemble=inference_ensemble,
                
            )   # already update the trajectory history/ids_to_results/current_id/id_deque in this function
            id_results = []
            for _ in ids:
                id_results.append(ids_to_results[_])
            id_results = torch.tensor(id_results, dtype=torch.long)
            if boxes_keep is not None:
                box_results = box_results[boxes_keep.cpu()]


        # Output to tracker file:
        if fake_submit is False:
            # Write the outputs to the tracker file:
            result_file_path = os.path.join(outputs_dir, "tracker", f"{seq_name}.txt")
            with open(result_file_path, "a") as file:
                assert len(id_results) == len(box_results), f"Boxes and IDs should in the same length, " \
                                                            f"but get len(IDs)={len(id_results)} and " \
                                                            f"len(Boxes)={len(box_results)}"
                for obj_id, box in zip(id_results, box_results):
                    obj_id = int(obj_id.item())
                    if len(box) < 4 :
                        box = box[0]
                    box = box*scale
                    x1, y1, x2, y2 = box.tolist()
                    if dataset in ["DanceTrack", "MOT17", "SportsMOT", "MOT17_SPLIT", "MOT15", "MOT15_V2"]:
                        result_line = f"{i + 1}," \
                                      f"{obj_id}," \
                                      f"{x1},{y1},{x2 - x1},{y2 - y1},1,-1,-1,-1\n"
                    else:
                        raise NotImplementedError(f"Do not know the outputs format of dataset '{dataset}'.")
                    file.write(result_line)

        images_first = record_images
        print(
            f"\rinference : [{i}/{len(seq_dataloader)}] ",
            end="")
    if fake_submit:
        print()
        print(f"[Fake] Finish >> Submit seq {seq_name.split('/')[-1]}. ")
    else:
        print()
        print(f"Finish >> Submit seq {seq_name.split('/')[-1]}. ")
    return

def get_seq_names(data_root: str, dataset: str, data_split: str):
    if dataset in ["DanceTrack", "SportsMOT", "MOT17", "MOT17_SPLIT"]:
        dataset_dir = os.path.join(data_root, dataset, data_split)
        return sorted(os.listdir(dataset_dir))
    else:
        raise NotImplementedError(f"Do not support dataset '{dataset}' for eval dataset.")


def get_eval_metrics_dict(metric_path: str):
    with open(metric_path) as f:
        metric_names = f.readline()[:-1].split(" ")
        metric_values = f.readline()[:-1].split(" ")
    metrics = {
        n: float(v) for n, v in zip(metric_names, metric_values)
    }
    return metrics

