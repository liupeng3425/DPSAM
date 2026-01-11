# Copyright by HQ-SAM team
# All rights reserved.

## data loader
from __future__ import print_function, division
import cv2

import numpy as np
import random
from copy import deepcopy
from skimage import io
import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

#### --------------------- dataloader online ---------------------####

def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
        tmp_im_list, tmp_gt_list = [], []
        tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])
        print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))

        if(datasets[i]["gt_dir"]==""):
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            tmp_gt_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
        print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))


        name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                "im_path":tmp_im_list,
                                "gt_path":tmp_gt_list,
                                "im_ext":datasets[i]["im_ext"],
                                "gt_ext":datasets[i]["gt_ext"]})
        
        if datasets[i]['depth_dir']:
            tmp_depth_list = []
            if 'SUNRGBD' in datasets[i]['name']:
                for item in tmp_im_list:
                    img_id = int(item.split('-')[-1].split('.')[0])
                    if '_MONO' in datasets[i]['name']:
                        img_id = item.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]
                        tmp_depth_list.append(os.path.join(datasets[i]['depth_dir'], f'{img_id}{datasets[i]["depth_ext"]}'))
                    else:
                        tmp_depth_list.append(os.path.join(datasets[i]['depth_dir'], f'{img_id}.png'))
                print('-depth-', datasets[i]["name"],datasets[i]["depth_dir"], ': ',len(tmp_depth_list))
                name_im_gt_list[-1]['depth_path'] = tmp_depth_list

            if 'NYUD' in datasets[i]['name']:
                for item in tmp_im_list:
                    img_id = item.split('/')[-1].split('.')[0]
                    tmp_depth_list.append(os.path.join(datasets[i]['depth_dir'], f'{img_id}{datasets[i]["depth_ext"]}'))
                print('-depth-', datasets[i]["name"],datasets[i]["depth_dir"], ': ',len(tmp_depth_list))
                name_im_gt_list[-1]['depth_path'] = tmp_depth_list
            
            if 'SCANNET' in datasets[i]['name']:
                for item in tmp_im_list:
                    img_id = item.split('/')[-1].split('.')[0]
                    tmp_depth_list.append(os.path.join(datasets[i]['depth_dir'], f'{img_id}{datasets[i]["depth_ext"]}'))
                print('-depth-', datasets[i]["name"],datasets[i]["depth_dir"], ': ',len(tmp_depth_list))
                name_im_gt_list[-1]['depth_path'] = tmp_depth_list
            if 'Stanford' in datasets[i]['name']:
                for item in tmp_im_list:
                    img_id = item.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]
                    tmp_depth_list.append(os.path.join(datasets[i]['depth_dir'], f'{img_id}{datasets[i]["depth_ext"]}'))
                print('-depth-', datasets[i]["name"],datasets[i]["depth_dir"], ': ',len(tmp_depth_list))
                name_im_gt_list[-1]['depth_path'] = tmp_depth_list
            

            if 'LLRGBD' in datasets[i]['name']:
                for item in tmp_im_list:
                    img_id = item.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]
                    tmp_depth_list.append(os.path.join(datasets[i]['depth_dir'], f'{img_id}{datasets[i]["depth_ext"]}'))
                print('-depth-', datasets[i]["name"],datasets[i]["depth_dir"], ': ',len(tmp_depth_list))
                name_im_gt_list[-1]['depth_path'] = tmp_depth_list
            if 'DELIVER_DEPTH' in datasets[i]['name']:
                for item in tmp_im_list:
                    img_id = item.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]
                    tmp_depth_list.append(os.path.join(datasets[i]['depth_dir'], f'{img_id}{datasets[i]["depth_ext"]}'))
                print('-depth-', datasets[i]["name"],datasets[i]["depth_dir"], ': ',len(tmp_depth_list))
                name_im_gt_list[-1]['depth_path'] = tmp_depth_list
            
            if 'SOD_' in datasets[i]['name']:
                for item in tmp_im_list:
                    img_id = item.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]
                    tmp_depth_list.append(os.path.join(datasets[i]['depth_dir'], f'{img_id}{datasets[i]["depth_ext"]}'))
                print('-depth-', datasets[i]["name"],datasets[i]["depth_dir"], ': ',len(tmp_depth_list))
                name_im_gt_list[-1]['depth_path'] = tmp_depth_list
            if 'DELIVER_LIDAR' in datasets[i]['name'] or 'KITTI360' in datasets[i]['name']:
                for item in tmp_im_list:
                    img_id = item.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]
                    tmp_depth_list.append(os.path.join(datasets[i]['depth_dir'], f'{img_id}{datasets[i]["depth_ext"]}'))
                print('-depth-', datasets[i]["name"],datasets[i]["depth_dir"], ': ',len(tmp_depth_list))
                name_im_gt_list[-1]['depth_path'] = tmp_depth_list
            
            if 'DELIVER_EVENT' in datasets[i]['name'] or 'EVENTSCAPE_EVENT' in datasets[i]['name']:
                for item in tmp_im_list:
                    img_id = item.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]
                    tmp_depth_list.append(os.path.join(datasets[i]['depth_dir'], f'{img_id}{datasets[i]["depth_ext"]}'))
                print('-depth-', datasets[i]["name"],datasets[i]["depth_dir"], ': ',len(tmp_depth_list))
                name_im_gt_list[-1]['depth_path'] = tmp_depth_list

            if 'DSEC_EVENT' in datasets[i]['name']:
                for item in tmp_im_list:
                    img_id = item.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]
                    tmp_depth_list.append(os.path.join(datasets[i]['depth_dir'], f'{img_id}{datasets[i]["depth_ext"]}'))
                print('-depth-', datasets[i]["name"],datasets[i]["depth_dir"], ': ',len(tmp_depth_list))
                name_im_gt_list[-1]['depth_path'] = tmp_depth_list
        
        if datasets[i]['name'] == 'SUNRGBD':
            name_im_gt_list = name_im_gt_list*5
        if datasets[i]['name'] == 'NYUDV2':
            name_im_gt_list = name_im_gt_list*5
        
        # if 'SCANNET' in datasets[i]['name']:
        #     name_im_gt_list = name_im_gt_list*2

    return name_im_gt_list

def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, training=False, prompt_val=False):
    gos_dataloaders = []
    gos_datasets = []

    if(len(name_im_gt_list)==0):
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if(batch_size>1):
        num_workers_ = 2
    if(batch_size>4):
        num_workers_ = 4
    if(batch_size>8):
        num_workers_ = 8


    if training:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms))
            gos_datasets.append(gos_dataset)

        gos_dataset = ConcatDataset(gos_datasets)
        sampler = DistributedSampler(gos_dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, batch_size, drop_last=True)
        dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)

        gos_dataloaders = dataloader
        gos_datasets = gos_dataset

    else:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True, prompt_val=prompt_val)
            sampler = DistributedSampler(gos_dataset, shuffle=False)
            dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)

            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets

class RandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        out_dict = sample
        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image,dims=[2])
            label = torch.flip(label,dims=[2])
            
            if sample['depth'] is not None:
                out_dict['depth'] = torch.flip(sample['depth'],dims=[2])
        out_dict.update({'imidx':imidx,'image':image, 'label':label, 'shape':shape})
        return out_dict

class Resize(object):
    def __init__(self,size=[320,320]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        out_dict = sample
        image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),self.size,mode='bilinear'),dim=0)
        label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),self.size,mode='nearest'),dim=0)
        if sample['depth'] is not None:
            depth = sample['depth']
            depth = torch.squeeze(F.interpolate(torch.unsqueeze(depth,0),self.size,mode='bilinear'),dim=0)
            out_dict['depth'] = depth
        
        out_dict.update({'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)})
        return out_dict
    
import matplotlib as mpl
class RenderDepth(object):
    def __init__(self,colormap='inferno'):
        self.colormap = colormap

    def __call__(self, sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        out_dict = sample
        if sample['depth'] is not None:
            depth = sample['depth'] # torch.Size([3, 480, 640])
            d = np.full(depth[0].shape, 0, dtype=float)
            d = (1 / (depth[0].numpy()+1e-6))
            colored_depth = (d - np.min(d)) / (np.max(d) - np.min(d))
            colored_depth = mpl.colormaps['inferno'](colored_depth)*255
            colored_depth = torch.tensor(colored_depth[:,:,:-1])
            colored_depth = colored_depth.permute(2,0,1).float()
                
            out_dict['depth'] = colored_depth
        return out_dict
    
class RandomCrop(object):
    def __init__(self,size=[288,288]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        out_dict = {}
        out_dict.update(sample)
        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:,top:top+new_h,left:left+new_w]
        label = label[:,top:top+new_h,left:left+new_w]
        if sample['depth'] is not None:
            depth = sample['depth']
            depth = depth[:,top:top+new_h,left:left+new_w]
            out_dict['depth'] = depth

        out_dict.update({'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)})
        return out_dict

class Normalize(object):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,sample):
        out_dict = {}
        out_dict.update(sample)

        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        image = normalize(image,self.mean,self.std)
        out_dict['image'] = image
        return out_dict



class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
        https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py 
    """

    def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, sample):

        out_dict = {}
        out_dict.update(sample)
        imidx, image, label, image_size =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        #resize keep ratio
        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().long()
        
        scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),scaled_size.tolist(),mode='nearest'),dim=0)
        
        if sample['depth'] is not None:
            depth = sample['depth']
            scaled_depth = torch.squeeze(F.interpolate(torch.unsqueeze(depth,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        # random crop
        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

        margin_h = max(scaled_size[0] - crop_size[0], 0).item()
        margin_w = max(scaled_size[1] - crop_size[1], 0).item()
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

        scaled_image = scaled_image[:,crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_label = scaled_label[:,crop_y1:crop_y2, crop_x1:crop_x2]
        if sample['depth'] is not None:
            scaled_depth = scaled_depth[:,crop_y1:crop_y2, crop_x1:crop_x2]

        # pad
        padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
        padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
        image = F.pad(scaled_image, [0,padding_w, 0,padding_h],value=128)
        label = F.pad(scaled_label, [0,padding_w, 0,padding_h],value=0)

        if sample['depth'] is not None:
            depth = F.pad(scaled_depth, [0,padding_w, 0,padding_h],value=1)
            out_dict['depth'] = depth

        out_dict.update({'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(image.shape[-2:])})


        return out_dict






class OnlineDataset(Dataset):
    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False, prompt_val=False):

        self.transform = transform
        self.dataset = {}
        ## combine different datasets into one
        dataset_names = []
        dt_name_list = [] # dataset name per image
        im_name_list = [] # image name
        im_path_list = [] # im path
        gt_path_list = [] # gt path
        im_ext_list = [] # im ext
        gt_ext_list = [] # gt ext
        depth_ext_list = [] # depth ext
        depth_path_list = [] # depth path
        for i in range(0,len(name_im_gt_list)):
            dataset_names.append(name_im_gt_list[i]["dataset_name"])
            # dataset name repeated based on the number of images in this dataset
            dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
            im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
            im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
            gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])
            
            depth_path_list.extend(name_im_gt_list[i].get('depth_path', ['']*len(name_im_gt_list[i]["gt_path"])))
            depth_ext_list.extend([item[-4:] for item in name_im_gt_list[i].get('depth_path', ['']*len(name_im_gt_list[i]["gt_path"]))])
                

        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list

        self.dataset['depth_path'] = depth_path_list
        self.dataset['depth_ext'] = depth_ext_list

        self.eval_ori_resolution = eval_ori_resolution
        self.prompt_val = prompt_val

    def __len__(self):
        return len(self.dataset["im_path"])

    def __getitem__(self, idx):
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]
        im = io.imread(im_path)
        gt = io.imread(gt_path)
        if 'Stanford' in self.dataset["data_name"][idx] and 'SEM' not in self.dataset["data_name"][idx]:
            gt = gt[:,:,0]
        if 'LLRGBD' in self.dataset["data_name"][idx]:
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if 'SOD_' in self.dataset["data_name"][idx]:
            gt = gt.astype(float)
        if self.dataset['depth_path'][idx] and self.dataset['depth_ext'][idx]:
            filename = self.dataset['depth_path'][idx]
            if 'SUNRGBD' in self.dataset["data_name"][idx]:
                depth = cv2.imread(str(filename), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)/100.
                if '_MONO' in self.dataset["data_name"][idx]:
                    depth = 255.0 / (depth + 0.01)
                else:
                    depth = 655.36 / (depth + 0.01)
                depth = torch.tensor(depth)
                if len(depth.shape)==2:
                    depth = torch.stack([depth, depth, depth], dim=0)

            elif 'SCANNET' in self.dataset["data_name"][idx]:
                depth = cv2.imread(str(filename), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)/100.
                depth = 655.36 / (depth + 0.01)
                depth = torch.tensor(depth)
                if len(depth.shape)==2:
                    depth = torch.stack([depth, depth, depth], dim=0)
            
            elif 'Stanford' in self.dataset["data_name"][idx]:
                depth = cv2.imread(str(filename), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)/100.
                depth = 655.36 / (depth + 0.01)
                depth = torch.tensor(depth)
                if len(depth.shape)==2:
                    depth = torch.stack([depth, depth, depth], dim=0)
            
            elif 'LLRGBD' in self.dataset["data_name"][idx]:
                if '_MONO' in self.dataset["data_name"][idx]:
                    depth = cv2.imread(str(filename), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
                    depth = 255.0 / (depth + 0.01)
                else:
                    depth = cv2.imread(str(filename), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)/100.
                    depth = 655.36 / (depth + 0.01)
                depth = torch.tensor(depth)
                if len(depth.shape)==2:
                    depth = torch.stack([depth, depth, depth], dim=0)
            elif 'SOD_' in self.dataset["data_name"][idx]:
                depth = cv2.imread(str(filename), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
                depth = 255.0 / (depth + 0.001)
                depth = torch.tensor(depth)
                if len(depth.shape)==2:
                    depth = torch.stack([depth, depth, depth], dim=0)
            
            elif 'NYUD' in self.dataset["data_name"][idx]:
                if '_MONO' in self.dataset["data_name"][idx]:
                    depth = cv2.imread(str(filename), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
                else:
                    depth = np.load(str(filename))
                depth = torch.tensor(depth)
                if len(depth.shape)==2:
                    depth = torch.stack([depth, depth, depth], dim=0)
            
            elif 'DELIVER_LIDAR' in self.dataset["data_name"][idx] or 'KITTI360' in self.dataset["data_name"][idx]:
                depth = cv2.imread(str(filename))
                depth = torch.tensor(depth).float()
                depth = depth.permute((2, 0, 1)).contiguous() / 100

            elif 'DELIVER_EVENT' in self.dataset["data_name"][idx] or 'EVENTSCAPE_EVENT' in self.dataset["data_name"][idx]:
                depth = cv2.imread(str(filename))
                depth = torch.tensor(depth).float()
                depth = depth.permute((2, 0, 1)).contiguous() / 255

            elif 'DSEC_EVENT' in self.dataset["data_name"][idx]:
                depth = cv2.imread(str(filename))
                depth = torch.tensor(depth).float()
                depth = depth.permute((2, 0, 1)).contiguous() / 255
            elif 'DELIVER_DEPTH' in self.dataset["data_name"][idx]:
                depth = cv2.imread(str(filename), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
                depth = 255.0 / (depth + 0.001)
                depth = torch.tensor(depth)
                if len(depth.shape)==2:
                    depth = torch.stack([depth, depth, depth], dim=0)
        else:
            depth = None
        
        if not self.prompt_val:
            # Prepare per-category binary masks
            if gt is not None and ('SUNRGBD' in self.dataset["data_name"][idx] or 'NYUD' in self.dataset["data_name"][idx]
                                or 'SCANNET' in self.dataset["data_name"][idx] 
                                or 'Stanford' in self.dataset["data_name"][idx] 
                                or 'LLRGBD' in self.dataset["data_name"][idx] or 'DELIVER' in self.dataset["data_name"][idx] 
                                or 'KITTI360' in self.dataset["data_name"][idx] or 'EVENTSCAPE' in self.dataset["data_name"][idx]
                                or 'DSEC_EVENT' in self.dataset["data_name"][idx] or 'DELIVER_DEPTH' in self.dataset["data_name"][idx]
                                ):
                sem_seg_gt = gt
                if len(sem_seg_gt.shape) > 2:
                    sem_seg_gt[:,:,0]
                classes = np.unique(sem_seg_gt)
                # remove ignored region
                classes = classes[classes != 0]
                if len(classes) == 0:
                    # Some image does not have annotation (all ignored)
                    new_id = random.choice(list(range(0,len(self))))
                    return self.__getitem__(new_id)

                gt = sem_seg_gt == random.choice(classes)

                    
            if len(gt.shape) > 2:
                gt = gt[:, :, 0]
            gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32),0)*255
        else:
            if len(gt.shape) > 2:
                gt = gt[:, :, 0]

            gt = torch.tensor(gt).unsqueeze(dim=0)
            if 'NJU' in self.dataset["data_name"][idx]:
                gt = gt.float()

        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        if 'DELIVER' in self.dataset["data_name"][idx] and im.shape[2] == 4:
            im = im[:,:,:3]
        im = torch.tensor(im.copy(), dtype=torch.float32)
        im = torch.transpose(torch.transpose(im,1,2),0,1)

        sample = {
            "imidx": torch.from_numpy(np.array(idx)),
            "image": im,
            "label": gt,
            "depth": depth,
            "shape": torch.tensor(im.shape[-2:]),
            'name': os.path.basename(im_path),
        }
        

        if self.transform:
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            sample["ori_label"] = gt.type(torch.uint8)  # NOTE for evaluation only. And no flip here
            sample['ori_im_path'] = self.dataset["im_path"][idx]
            sample['ori_gt_path'] = self.dataset["gt_path"][idx]

        return sample