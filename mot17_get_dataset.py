import json
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import cv2
# from models.embedding import EmbeddingComputer
import pickle
from models.feature_embedding import SimpleCNN
from torchvision import transforms as T

transform = T.Compose([
    T.Resize((256, 128)),   # 将图像调整为固定尺寸
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

def mkdirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def is_continuous(column):
    """
    判断轨迹是否连续
    :param column:
    :return:
    """
    unique_values = np.unique(column)
    return np.all(np.diff(unique_values) == 1)


def get_splits(mot_train_dir, splits=None):
    """
    Get the scene names for the given splits (train_half, val_half).
    """
    if splits is None:
        splits = ['train_half', 'val_half']

    split_scenes = {'train_half': [], 'val_half': [], 'videos': []}

    # Check if the split files exist, read from them
    for split in splits:
        split_path = os.path.join(mot_train_dir, f"annotations/{split}.json")
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                data = json.load(f)
                scenes = [video['file_name'] for video in data["images"]]
                split_scenes[split] = scenes
                split_scenes["videos"] = [video['file_name'] for video in data["videos"]]

    return split_scenes


def trajectory_interpolation(data):
    """
    对tid的gt轨迹的缺失进行插值
    :param data:
    :return:
    """
    df = pd.DataFrame(data, columns=['frame_index', 'track_id', 'x1', 'y1', 'width', 'height', 'vis'])
    df = df.sort_values(by='frame_index')
    full_index = np.arange(df['frame_index'].min(), df['frame_index'].max() + 1)
    complete_df = pd.DataFrame({'frame_index': full_index})
    complete_df = complete_df.merge(df, on='frame_index', how='left')
    complete_df[['track_id', 'x1', 'y1', 'width', 'height']] = complete_df[
        ['track_id', 'x1', 'y1', 'width', 'height']].interpolate()
    complete_df.fillna(value=0.0, inplace=True)
    return complete_df.to_numpy()

def sample_dancetrack(dataset_root, seq_len, save_root):
    # Get the scenes for train_half and val_half
    split_scenes = get_splits(dataset_root, splits=['train', 'val'])
    # embedding = EmbeddingComputer("mot17", False, grid_off=True)
    path = r"F:\GIP\external\weights\dance_sbs_S50.pth"
    model = SimpleCNN().to('cuda:4').eval()
    print(f'Start processing datasets')
    mkdirs(save_root + '/embs')

    seq_root = os.path.join(dataset_root, "train")
    seqs = [s for s in os.listdir(seq_root) if os.path.isdir(os.path.join(seq_root, s))]  # 视频序列名list

    train_sample_dataset = []
    val_sample_dataset = []
    for seq in seqs:
        # if seq not in split_scenes['videos']:
        #     continue
        print(seq + " : ")
        seq_info = open(os.path.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

        gt_txt = os.path.join(seq_root, seq, 'gt', 'gt.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
        # MOT17格式: [frame_id, track_id, x, y, w, h, conf, class_id, visibility]
        gt = gt[gt[:, 6] >= 0]  # 保留置信度>=0的目标
        gt = gt[gt[:, 7] == 1]  # 只保留行人类别(class_id=1)
        # fid, tid, x, y, w, h, vis
        gt = gt[:, [0, 1, 2, 3, 4, 5, 8]]

        # 提取所有帧的目标嵌入
        for item in gt:
            fid, tid = item[0], item[1]
            bbox = item[2:6]
            # 检查bbox的合法性
            if bbox[2] <= 0 or bbox[3] <= 0:  # 宽度或高度小于等于0
                continue
            if bbox[0] < 0 or bbox[1] < 0:  # 左上角坐标为负
                if bbox[0] < 0:
                    bbox[0] = 0
                if bbox[1] < 0:
                    bbox[1] = 0
            #if bbox[0] + bbox[2] > seq_width or bbox[1] + bbox[3] > seq_height:  # 超出图像边界
            #    continue
            # convert x1y1wh to xyxy
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            tag = f'{seq}_{int(fid)}_{int(tid)}'
            emb_path = f'{save_root}/embs/{tag}.pkl'
            if os.path.exists(emb_path):
                continue
            else:
                # 读取图片并计算embedding
                img_path = f'{seq_root}/{seq}/img1/{int(fid):08}.jpg'
                img = cv2.imread(img_path)
                x1, y1, x2, y2 = map(int, bbox)
                crop = img[y1:y2,x1:x2,:]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                normalize = 'True'
                if normalize:
                    crop /= 255
                    crop -= np.array((0.485, 0.456, 0.406))
                    crop /= np.array((0.229, 0.224, 0.225))
                crop = torch.as_tensor(crop.transpose(2, 0, 1))
                crop = crop.unsqueeze(0).to('cuda:4')
                with torch.no_grad():
                    embeddings = model(crop).float()

                pickle.dump(embeddings, open(emb_path, 'wb'))





def sample_MOT17(dataset_root, seq_len, save_root):
    # Get the scenes for train_half and val_half
    split_scenes = get_splits(dataset_root, splits=['train_half', 'val_half'])
    # embedding = EmbeddingComputer("mot17", False, grid_off=True)

    print(f'Start processing datasets')
    mkdirs(save_root + '/embs')

    seq_root = os.path.join(dataset_root, "train")
    seqs = [s for s in os.listdir(seq_root) if os.path.isdir(os.path.join(seq_root, s))]  # 视频序列名list

    train_sample_dataset = []
    val_sample_dataset = []
    for seq in seqs:
        if seq not in split_scenes['videos']:
            continue
        print(seq + " : ")
        seq_info = open(os.path.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

        gt_txt = os.path.join(seq_root, seq, 'gt', 'gt.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
        # MOT17格式: [frame_id, track_id, x, y, w, h, conf, class_id, visibility]
        gt = gt[gt[:, 6] >= 0]  # 保留置信度>=0的目标
        gt = gt[gt[:, 7] == 1]  # 只保留行人类别(class_id=1)
        # fid, tid, x, y, w, h, vis
        gt = gt[:, [0, 1, 2, 3, 4, 5, 8]]

        # 提取所有帧的目标嵌入
        for item in gt:
            fid, tid = item[0], item[1]
            bbox = item[2:6]
            # 检查bbox的合法性
            if bbox[2] <= 0 or bbox[3] <= 0:  # 宽度或高度小于等于0
                continue
            if bbox[0] < 0 or bbox[1] < 0:  # 左上角坐标为负
                continue
            #if bbox[0] + bbox[2] > seq_width or bbox[1] + bbox[3] > seq_height:  # 超出图像边界
            #    continue
            # convert x1y1wh to xyxy
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            tag = f'{seq}_{int(fid)}_{int(tid)}'
            emb_path = f'{save_root}/embs/{tag}.pkl'
            if os.path.exists(emb_path):
                continue
            else:
                # 读取图片并计算embedding
                img_path = f'{seq_root}/{seq}/img1/{int(fid):06}.jpg'
                img = cv2.imread(img_path)
                embeddings = embedding.compute_embedding(img, [bbox], tag)
                pickle.dump(embeddings, open(emb_path, 'wb'))

        # 转换为中心点坐标并归一化
        gt[:, 2] += gt[:, 4] / 2  # x + w/2 -> cx
        gt[:, 3] += gt[:, 5] / 2  # y + h/2 -> cy
        # 归一化坐标
        gt[:, 2] /= seq_width  # cx
        gt[:, 4] /= seq_width  # w
        gt[:, 3] /= seq_height  # cy
        gt[:, 5] /= seq_height  # h

        num_tids = int(max(gt[:, 1]))  # seq中的轨迹数量
        seq_tid_gt = {}  # 存储插值后不同tid的轨迹 {tid : tid_gt}
        for tid in range(0, num_tids + 1):
            tid_gt = gt[gt[:, 1] == tid]
            if len(tid_gt) == 0:  # 跳过不存在的轨迹ID
                continue
            # 对不连续帧进行插值
            if not is_continuous(tid_gt[:, 0]):
                tid_gt = trajectory_interpolation(tid_gt)
            seq_tid_gt[tid] = tid_gt

        max_frame = int(max(gt[:, 0]))
        # 每次取连续的6帧
        for start_frame in range(1, max_frame - 4):
            frame_data = []
            frame_tags = []
            # 收集连续6帧的数据
            for frame_idx in range(start_frame, start_frame + seq_len):
                # 获取当前帧的所有目标
                frame_gt = gt[gt[:, 0] == frame_idx]
                if len(frame_gt) > 0:
                    # 收集当前帧所有目标的位置信息
                    frame_boxes = frame_gt[:, 2:6]  # cx, cy, w, h
                    # 收集当前帧所有目标的embedding tags
                    frame_obj_tags = [f'{seq}_{frame_idx}_{int(tid)}' for tid in frame_gt[:, 1]]
                    frame_data.append(frame_boxes)
                    frame_tags.append(frame_obj_tags)

            # 确保获取到了连续6帧的数据
            if len(frame_data) == seq_len:
                sample_item = {
                    'bbox': frame_data,  # list of (N, 4) arrays，N是每帧中目标数量
                    'tags': frame_tags,  # list of lists，每个子列表包含当前帧所有目标的embedding tags
                }
                if (seq + f"/img1/{start_frame:06d}.jpg") in split_scenes["val_half"]:
                    train_sample_dataset.append(sample_item)
                else:
                    val_sample_dataset.append(sample_item)
    # 保存train_sample_dataset
    with open(os.path.join(save_root, f'train_mot17_{seq_len}.pkl'), 'wb') as f:
        pickle.dump(train_sample_dataset, f)
    # 保存val_sample_dataset
    with open(os.path.join(save_root, f'val_mot17_{seq_len}.pkl'), 'wb') as f:
        pickle.dump(val_sample_dataset, f)


if __name__ == "__main__":
    dataset_root = "/home1/lcy/FIP-v3/datasets/DanceTrack/"
    save_root = 'sample_datasets/dancetrack'
    seq_len = 5
    sample_dancetrack(dataset_root, seq_len, save_root)
