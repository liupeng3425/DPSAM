from glob import glob
import os
import shutil

from tqdm import tqdm

def copy_files(src_path, dst_path, file_list):
    os.makedirs(os.path.join(dst_path, 'color'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'label'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'instance'), exist_ok=True)
    os.makedirs(os.path.join(dst_path, 'pose'), exist_ok=True)
    for line in tqdm(file_list):
        scan_id = line.strip()
        for item in glob(os.path.join(src_path,scan_id,'color','*.jpg')):
            fname = os.path.basename(item).split('.')[0]
            shutil.copy(item, os.path.join(dst_path, 'color', f'{scan_id}_{fname}.jpg'))
            shutil.copy(item.replace('color', 'depth').replace('.jpg', '.png'), os.path.join(dst_path, 'depth', f'{scan_id}_{fname}.png'))
            shutil.copy(item.replace('color', 'instance').replace('.jpg', '.png'), os.path.join(dst_path, 'instance', f'{scan_id}_{fname}.png'))
            shutil.copy(item.replace('color', 'label').replace('.jpg', '.png'), os.path.join(dst_path, 'label', f'{scan_id}_{fname}.png'))
            shutil.copy(item.replace('color', 'pose').replace('.jpg', '.txt'), os.path.join(dst_path, 'pose', f'{scan_id}_{fname}.txt'))

def main():
    train_file = '/data1/liupeng/sam-hq/train/data/scannetv2_train.txt'
    val_file = '/data1/liupeng/sam-hq/train/data/scannetv2_val.txt'
    test_file = '/data1/liupeng/sam-hq/train/data/scannetv2_test.txt'
    src_path = '/data1/liupeng/sam-hq/train/data/scannet_frames_25k'
    train_dst_path = '/data1/liupeng/sam-hq/train/data/scannet_frames_25k_train'
    val_dst_path = '/data1/liupeng/sam-hq/train/data/scannet_frames_25k_val'
    test_dst_path = '/data1/liupeng/sam-hq/train/data/scannet_frames_25k_test'

    with open(train_file, 'r') as f:
        train_list = f.readlines()
    with open(val_file, 'r') as f:
        val_list = f.readlines()
    with open(test_file, 'r') as f:
        test_list = f.readlines()

    copy_files(src_path, train_dst_path, train_list)
    copy_files(src_path, val_dst_path, val_list)
    copy_files(src_path, test_dst_path, test_list)

if __name__ == '__main__':
    main()
