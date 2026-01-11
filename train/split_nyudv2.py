import os
import shutil

def copy_files(src_path, dst_path, file_list):
    for line in file_list:
        rgb_path, label_path = line.strip().split('\t')
        rgb_path = os.path.join(src_path, rgb_path)
        label_path = os.path.join(src_path, label_path)
        os.makedirs(os.path.join(dst_path, 'RGB'), exist_ok=True)
        os.makedirs(os.path.join(dst_path, 'Label'), exist_ok=True)
        os.makedirs(os.path.join(dst_path, 'HHA'), exist_ok=True)
        os.makedirs(os.path.join(dst_path, 'Depth'), exist_ok=True)
        shutil.copy(rgb_path, os.path.join(dst_path, 'RGB'))
        shutil.copy(label_path, os.path.join(dst_path, 'Label'))
        shutil.copy(rgb_path.replace('RGB', 'HHA').replace('jpg', 'jpg'), os.path.join(dst_path, 'HHA'))
        shutil.copy(rgb_path.replace('RGB', 'Depth').replace('jpg', 'npy'), os.path.join(dst_path, 'Depth'))

def main():
    train_file = '/data1/liupeng/sam-hq/train/data/NYUDepthv2/train.txt'
    test_file = '/data1/liupeng/sam-hq/train/data/NYUDepthv2/test.txt'
    src_path = '/data1/liupeng/sam-hq/train/data/NYUDepthv2'
    train_dst_path = '/data1/liupeng/sam-hq/train/data/NYUDepthv2train'
    test_dst_path = '/data1/liupeng/sam-hq/train/data/NYUDepthv2test'

    with open(train_file, 'r') as f:
        train_list = f.readlines()
    with open(test_file, 'r') as f:
        test_list = f.readlines()

    copy_files(src_path, train_dst_path, train_list)
    copy_files(src_path, test_dst_path, test_list)

if __name__ == '__main__':
    main()
