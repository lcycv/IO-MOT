import os
import shutil


def is_directory_empty(path):
    return len(os.listdir(path)) == 0


def copy_files(src, dst):

    if os.path.isdir(src):
        for item in os.listdir(src):
            item_path = os.path.join(src, item)
            copy_files(item_path, os.path.join(dst, item))
    else:

        shutil.copy2(src, dst)


save_path = 'train'
path = r"F:\data\BFT\annotations_mot\train"
img_path = r'F:\data\BFT\train_image'
gt_path = r'F:\data\BFT\annotations_mot\train'
for i in os.listdir(path):

    i = i.split(".")[0]


    target_img_dir = os.path.join(save_path, i, "img1")
    target_gt_dir = os.path.join(save_path, i, "gt")
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_gt_dir, exist_ok=True)


    if is_directory_empty(target_img_dir):
        src_img_dir = os.path.join(img_path, i)
        copy_files(src_img_dir, target_img_dir)  # 递归复制文件夹内容

    if is_directory_empty(target_gt_dir):
        src_img_dir = os.path.join(gt_path, i+".txt")
        copy_files(src_img_dir, target_gt_dir)  # 递归复制文件夹内容


    if 'seqinfo.ini' not in os.listdir(os.path.join(save_path, i)):
        import configparser
        from PIL import Image


        config = configparser.ConfigParser()

        example_img_path = os.listdir(target_img_dir)[0]
        with Image.open(os.path.join(target_img_dir,example_img_path)) as img:

            width, height = img.size


        config['Sequence'] = {
            'name': i,
            'imDir': 'img1',
            'frameRate': '20',
            'seqLength': len(os.listdir(img_path)),
            'imWidth': width,
            'imHeight': height,
            'imExt': '.jpg'
        }


        with open(os.path.join(save_path, i,'seqinfo.ini'), 'w') as configfile:
            config.write(configfile)
