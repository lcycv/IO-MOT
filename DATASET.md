# Data Preparation

:link: For all the datasets we used in our experiments, you can access them from the following public link:
- Get the [detection_results](https://github.com/Kroery/DiffMOT/releases/download/v1.1/Detections.zip) 
- [DanceTrack](https://github.com/DanceTrack/DanceTrack)
- [SportsMOT](https://github.com/MCG-NJU/SportsMOT)
- [BFT](https://george-zhuang.github.io/nettrack/)


## Generate GT files

For the BFT  datasets, you’ll need to use the provided script to convert their ground truth files to the format we require:

- For BFT: [create_bft.py](/data/create_bft.py)

:pushpin: You need to modify the paths in the script according to your requirements.

## File Tree

```text
<DATADIR>/
  ├── DanceTrack/
  │ ├── train/
  │ ├── val/
  │ ├── test/
  │ ├── detections_yolox_x/
  │ ├── train_seqmap.txt
  │ ├── val_seqmap.txt
  │ └── test_seqmap.txt
  ├── SportsMOT/
  │ ├── train/
  │ ├── val/
  │ ├── test/
  │ ├── detections_yolox_x/
  │ ├── train_seqmap.txt
  │ ├── val_seqmap.txt
  │ └── test_seqmap.txt
  └── BFT/
   ├── train/
   ├── val/
   ├── test/
   ├── detections_yolox_x/
   ├── train_seqmap.txt
   ├── val_seqmap.txt
   └── test_seqmap.txt
 
    
```
