# IO-MOT: Dual-Modal Feature Fusion with Instance Embeddings and Optical Flow for Multi-Object Tracking  

🚀 **A novel framework combining instance embeddings and optical flow for robust multi-object tracking.**  

---


| Dataset        | HOTA | DETA | ASSA| MOTA | IDF1|                              Result                              |                                    Weight                               |          
| :------------- | :--: | :--: | :--: | :--: | :--: | :-----------------------------------------------------------------: |  :-----------------------------------------------------------------: | 
| **DanceTrack** | 67.5| 81.5|56.0 | 91.0 |69.9 |[📥 download](https://drive.google.com/file/d/1vytRYhRMRk5Lvat7W0QxRob9Wx2P0iDr/view?usp=drive_link)|[📥 download](https://drive.google.com/file/d/1xKwtflVf3HWfpLMyh5bD_J9_7kgE6zjR/view?usp=drive_link) |  
| **SportsMOT**  | 73.2 | 84.9 |  63.1 | 94.2 |  75.9 |[📥 download](https://drive.google.com/file/d/1UeNmOaN0qrZ0IItXpWmiSZoyGq0px2aj/view?usp=drive_link)|  [📥 download](https://drive.google.com/file/d/1EKG_UQMdQPZQtnGB4H9cZviBhzfn96Ih/view?usp=drive_link)| 
| **BFT**     |  68.6 |   67.3  |  70.0 |   73.3  |  80.2|[📥 download](https://drive.google.com/file/d/1mgG_jK-oea1UYf_ACZXu0Zp-pv8BT4GJ/view?usp=drive_link) |   [📥 download](https://drive.google.com/file/d/1mgG_jK-oea1UYf_ACZXu0Zp-pv8BT4GJ/view?usp=drive_link)   | 
---
🔗 **Detection Results & FastReID Pretrained Weights:** [Google Drive](https://drive.google.com/drive/folders/1HViyb73bdv4ZT05sv7kKAygIyu3kNDU8?usp=drive_link)  


## 📌 Quick Start  
## Instructions
```shell
conda create -n IO-MOT python=3.11		# suggest to use virtual envs
conda activate IO-MOT
# PyTorch:
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia		# CUDA version=12.1 is also OK
# Other dependencies:
conda install matplotlib pyyaml scipy tqdm tensorboard seaborn scikit-learn pandas
pip install opencv-python einops pycocotools timm
```
Install [FastFlowNet](https://github.com/ltkong218/FastFlowNet) \
Install FastReID( you can follow the work [DiffMOT](https://github.com/Kroery/DiffMOT) )


### **1. Training**  
```bash
python -m torch.distributed.run --nproc_per_node=8 main.py \
    --mode train \
    --use-distributed True \
    --config-path <your_config_path> \
    --data-root <your_data_path> \
    --outputs-dir <save_dir>
```


### **2. Inference**  
```bash
python main.py \
    --mode eval \
    --config-path <your_config_path> \
    --data-root <your_data_path> \
    --inference-model <model_path> \
    --outputs-dir <save_dir> \
    --inference-dataset <dataset_name>
```
