# Copyright (c) RuopengGao. All Rights Reserved.
# About:
import os
import cv2
from collections import defaultdict
import torchvision.transforms.functional as F

from torch.utils.data import Dataset

def default_list():
    return list()

def default_dict():
    return defaultdict(default_list)

def ddefault_dict():
    return defaultdict(default_dict)

def dddefault_dict():
    return defaultdict(ddefault_dict)



class SeqDataset(Dataset):
    def __init__(self, seq_dir: str, dataset: str, height: int = 800, width: int = 1333):
        """
        Args:
            seq_dir:
            dataset: DanceTrack or MOT17 or et al.
        """
        image_paths = sorted(os.listdir(os.path.join(seq_dir, "img1")))
        image_paths = [os.path.join(seq_dir, "img1", _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
        self.image_paths = image_paths
        self.image_height = height
        self.image_width = width
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.dataset = dataset
        det_path = os.path.join(seq_dir, "gt/gt.txt")

        use_gt = False

        # 使用普通函数构造默认字典
        infos = dddefault_dict()
        for f in range(len(image_paths)):
            infos[dataset][f]["prob"] = []
            infos[dataset][f]["box"] = []


        if use_gt == False:
            seq_name = seq_dir.split("/")
            det_path = os.path.join(seq_name[1],seq_name[2],"detections_yolox_x",seq_name[3],seq_name[4])
            index = None
            if dataset == "DanceTrack" or dataset == "SportsMOT":
                for frame in sorted(os.listdir(det_path)):

                    with open(os.path.join(det_path ,frame), "r") as det_file:
                        # [frame,  x, y, w, h, prob, ]
                        for line in det_file:
                            
                            f, x, y, w, h, v= line.split(",")
                   
                            f, = map(int, (f,))
                            if index == None:  #检测器输出格式问题
                                index = f-1
                            x, y, w, h, v = map(float, (x, y, w, h, v))

                            infos[dataset][f-index]["prob"].append([
                                float(v)
                            ])
                            infos[dataset][f-index]["box"].append([
                                float(x), float(y), float(w), float(h)
                            ])
                            pass
            elif dataset == 'MOT17' or dataset == "MOT17_SPLIT" :
                for frame in sorted(os.listdir(det_path)):

                    with open(os.path.join(det_path ,frame), "r") as det_file:
                        # [frame,  x, y, w, h, prob, ]
                        for line in det_file:
                            f, x, y, w, h, v= line.split(",")
                            f, = map(int, (f,))
                            if index == None:  #检测器输出格式问题
                                index = f-1
                            x, y, w, h, v = map(float, (x, y, w, h, v))

                            infos[dataset][f-index]["prob"].append([
                                float(v)
                            ])
                            infos[dataset][f-index]["box"].append([
                                float(x), float(y), float(w), float(h)
                            ])
                            pass

        if use_gt == True:
            if det_path is not None:
                with open(det_path, "r") as gt_file:
                    for line in gt_file:
                        line = line[:-1]
                        if dataset == "DanceTrack" or dataset == "SportsMOT":
                            # [frame, id, x, y, w, h, 1, 1, 1]
                            f, i, x, y, w, h, _, _, _ = line.split(",")
                            label = 0
                            v = 1
                        elif dataset == "MOT17" or dataset == "MOT17_SPLIT" or dataset == "Visdrone" :
                            f, i, x, y, w, h, v = line.split(" ")
                            label = 0
                        elif dataset == "CrowdHuman":
                            f, i, x, y, w, h = line.split(" ")
                            label = 0
                            v = 1
                        else:
                            raise NotImplementedError(f"Can't analysis the gts of dataset '{dataset}'.")
                        # format, and write into infos
                        f, i, label = map(int, (f, i, label))
                        x, y, w, h, v = map(float, (x, y, w, h, v))
                        # assert v != 0.0, f"Visibility of object '{i}' in frame '{f}' is 0.0."
                        infos[dataset][f]["prob"].append([
                            float(v)
                        ])
                        infos[dataset][f]["box"].append([
                            float(x),  float(y), float(w),  float(h)
                        ])
                        pass
            else:
                assert 0
            pass
        self.gt_info = infos


        return

    @staticmethod
    def load(path):
        """
        Args:
            path:

        Returns:
        """
        # label_path = path.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
        image = cv2.imread(path)
        assert image is not None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def process_image(self, image, item):
        ori_image = image.copy()
        h, w = image.shape[:2]
        scale = self.image_height / min(h, w)
        if max(h, w) * scale > self.image_width:
            scale = self.image_width / max(h, w)
        target_h = int(h * scale)
        target_w = int(w * scale)
        image = cv2.resize(image, (target_w, target_h))

        # image = F.to_tensor(image)

        image = F.normalize(F.to_tensor(image), self.mean, self.std)

        # image = image.unsqueeze(0)
        box = self.gt_info[self.dataset][item+1]["box"]
        prob = self.gt_info[self.dataset][item+1]["prob"]
        for i, ( x, y, w, h) in enumerate(box):
            # 计算新的缩放后的值
            x = (x * scale)
            y = (y * scale)
            w = (w * scale)
            h = (h * scale)

            # 更新 det 列表中的目标框
            box[i] = [ x, y, w, h]

        return image, ori_image, prob, box

    # def process_image(self, image, item):
    #     ori_image = image.copy()
    #     h, w = image.shape[:2]
    #     scale = self.image_height / min(h, w)
    #     if max(h, w) * scale > self.image_width:
    #         scale = self.image_width / max(h, w)
    #     target_h = int(h * scale)
    #     target_w = int(w * scale)
    #     image = cv2.resize(image, (target_w, target_h))
    #     image = F.normalize(F.to_tensor(image), self.mean, self.std)
    #     # image = image.unsqueeze(0)
    #     ratio_w = target_w / w
    #     ratio_h = target_h / h
    #     det = self.gt_info[self.dataset][item + 1]["box"]
    #     box = torch.tensor(det) * torch.as_tensor([ratio_w, ratio_h, ratio_w, ratio_h])
    #
    #     prob = torch.tensor(self.gt_info[self.dataset][item + 1]["prob"])
    #     # for i, (v, x, y, w, h) in enumerate(det):
    #     #     # 计算新的缩放后的值
    #     #     x = int(x * scale)
    #     #     y = int(y * scale)
    #     #     w = int(w * scale)
    #     #     h = int(h * scale)
    #     #
    #     #     # 更新 det 列表中的目标框
    #     #     det[i] = (v, x, y, w, h)
    #
    #     return image, ori_image, prob, box, det


    def __getitem__(self, item):
        image = self.load(self.image_paths[item])
        return self.process_image(image=image, item=item)

    def __len__(self):
        return len(self.image_paths)

if __name__=='__main__':
    img=cv2.imread(r'F:\GIP\datasets\DanceTrack\train\dancetrack0001\img1\00000001.jpg')

    print(img)