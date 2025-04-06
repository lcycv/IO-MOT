IO-MOT: Dual-Modal Feature Fusion with Instance Embeddings and Optical Flow for  Multi-Object Tracking

All our training scripts follow the template script below. You'll need to fill the <placeholders> according to your requirements：
For training：
python -m torch.distributed.run --nproc_per_node=8 main.py --mode train --use-distributed True --config-path <config file path> --data-root <DATADIR> --outputs-dir <outputs dir>

For inference：
python  main.py --mode eval   --config-path <config file path> --data-root <DATADIR> --inference-model <checkpoint path> --outputs-dir <outputs dir> --inference-dataset <dataset name> 

