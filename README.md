# IO-MOT: Dual-Modal Feature Fusion with Instance Embeddings and Optical Flow for Multi-Object Tracking  

🚀 **A novel framework combining instance embeddings and optical flow for robust multi-object tracking.**  

---

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
