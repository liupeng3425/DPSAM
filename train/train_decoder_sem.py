# Copyright by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Any, Dict, List, Set, Tuple

from segment_anything_training.build_sam import sam_model_registry
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
import utils.misc as misc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096"

def show_anns(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())

        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)

    parser.add_argument("--output", type=str, required=True, 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_l", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=12, type=int)
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--batch_size_train', default=4, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")

    return parser.parse_args()


def main(net, train_datasets, valid_datasets, args):

    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    net = sam_model_registry[args.model_type](checkpoint=args.checkpoint)

    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                        my_transforms = [
                                                                    RandomHFlip(),
                                                                    LargeScaleJitter()
                                                                    ],
                                                        batch_size = args.batch_size_train,
                                                        training = True)
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                          my_transforms = [
                                                                        Resize(args.input_size)
                                                                    ],
                                                          batch_size=args.batch_size_valid,
                                                          training=False)
    print(len(valid_dataloaders), " valid dataloaders created")
    
    ### --- Step 2: DistributedDataParallel---
    if torch.cuda.is_available():
        net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    net_without_ddp = net.module

 
    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
        print("--- define optimizer ---")
        defaults = {}
        defaults["lr"] = args.learning_rate
        defaults["weight_decay"] = 0
        defaults["betas"]=(0.9, 0.999)
        defaults["eps"]=1e-08

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in net_without_ddp.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                if "prompt_encoder" in module_name:
                    continue
                # if "mask_decoder" in module_name:
                #     continue

                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                # params.append({"params": [value], **hyperparams})
                params.append({'name':f'{module_name}.{module_param_name}',"params": [value], **hyperparams})

        optimizer = optim.Adam(params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        # optimizer = optim.Adam(net_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch

        train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
    else:
        # sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        # _ = sam.to(device=args.device)
        # sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)

        if args.restore_model:
            print("restore model from:", args.restore_model)
            if torch.cuda.is_available():
                net_without_ddp.load_state_dict(torch.load(args.restore_model))
            else:
                net_without_ddp.load_state_dict(torch.load(args.restore_model,map_location="cpu"))
    
        evaluate(args, net, None, valid_dataloaders, args.visualize)


def train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)

    net.train()
    _ = net.to(device=args.device)
    
    best_iou = 0
    for epoch in range(epoch_start,epoch_num): 
        iter_num = 0
        print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)

        for data in metric_logger.log_every(train_dataloaders,10):
            # iter_num = iter_num + 1
            # if iter_num> 100:
            #     break
            inputs, labels = data['image'], data['label']
            depths = data['depth']
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
                depths = depths.cuda()

            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
            
            # input prompt
            # input_keys = ['box','point','noise_mask']
            input_keys = ['point']
            labels_box = misc.masks_to_boxes(labels[:,0,:,:])
            try:
                kp = random.choice([1,2,3,5,10])
                labels_points = misc.masks_sample_points(labels[:,0,:,:],k=kp)
            except:
                # less than 10 points
                input_keys = ['box']#,'noise_mask']
            labels_256 = F.interpolate(labels, size=(256, 256), mode='nearest')
            labels_noisemask = misc.masks_noise(labels_256)

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=net.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                dict_input['depth'] = depths[b_i]
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            # with torch.no_grad():
            batched_output, interm_embeddings = net(batched_input, multimask_output=False)
            mask_sam = torch.cat([item['low_res_logits'] for item in batched_output], dim=0)
            # mask_sam = nn.Sigmoid()(mask_sam)
            
            loss_mask, loss_dice = loss_masks(mask_sam, labels/255, len(mask_sam))
            loss = loss_mask + loss_dice
            
            loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice}

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)
            # if epoch==0:
            #     break


        print("Finished epoch:      ", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        lr_scheduler.step()
        test_stats = evaluate(args, net, None, valid_dataloaders)
        if misc.is_main_process() and best_iou<test_stats['val_iou_0']:
            best_iou = test_stats['val_iou_0']
            best_name = epoch
        print('best epoch is ', epoch, 'best iou ', best_iou)
        train_stats.update(test_stats)
        
        net.train()  

        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_"+str(epoch)+".pth"
            print('come here save at', args.output + model_name)
            misc.save_on_master(net.module.state_dict(), args.output + model_name)
    
    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    
    # merge sam and hq_decoder
    if misc.is_main_process():
        sam_ckpt = torch.load(args.checkpoint)
        hq_decoder = torch.load(args.output + model_name)
        for key in hq_decoder.keys():
            sam_key = 'mask_decoder.'+key
            if sam_key not in sam_ckpt.keys():
                sam_ckpt[sam_key] = hq_decoder[key]
        model_name = "/sam_hq_epoch_"+str(epoch)+".pth"
        torch.save(sam_ckpt, args.output + model_name)
        print('best epoch is ', epoch, 'best iou ', best_iou)



def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)

def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],postprocess_preds[i])
    return iou / len(preds)

def evaluate(args, net, sam, valid_dataloaders, visualize=False):
    net.eval()
    print("Validating...")
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        for data_val in metric_logger.log_every(valid_dataloader,100):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label']
            depth_val = data_val['depth']
            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                labels_ori = labels_ori.cuda()
                depth_val = depth_val.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
            
            labels_box = misc.masks_to_boxes(labels_val[:,0,:,:])
            input_keys = ['point']

            try:
                kp = random.choice([1,2,3,5,10])
                labels_points = misc.masks_sample_points(labels_val[:,0,:,:], k=kp)
            except:
                # less than 10 points
                input_keys = ['box','noise_mask']
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=net.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                dict_input['depth'] = depth_val[b_i] 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                batched_output, interm_embeddings = net(batched_input, multimask_output=False)
            
            
            mask_sam = torch.cat([item['low_res_logits'] for item in batched_output], dim=0)

            iou = compute_iou(mask_sam,labels_ori)
            boundary_iou = compute_boundary_iou(mask_sam,labels_ori)

            if visualize:
                print("visualize")
                os.makedirs(args.output, exist_ok=True)
                masks_hq_vis = (F.interpolate(mask_sam.detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()
                for ii in range(len(imgs)):
                    base = data_val['imidx'][ii].item()
                    print('base:', base)
                    save_base = os.path.join(args.output, str(k)+'_'+ str(base))
                    imgs_ii = imgs[ii].astype(dtype=np.uint8)
                    show_iou = torch.tensor([iou.item()])
                    show_boundary_iou = torch.tensor([boundary_iou.item()])
                    show_anns(masks_hq_vis[ii], None, labels_box[ii].cpu(), None, save_base , imgs_ii, show_iou, show_boundary_iou)
                       

            loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)


        print('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)


    return test_stats


if __name__ == "__main__":

    ### --------------- Configuring the Train and Valid datasets ---------------

    dataset_dis = {"name": "DIS5K-TR",
                 "im_dir": "./data/DIS5K/DIS-TR/im",
                 "gt_dir": "./data/DIS5K/DIS-TR/gt",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_thin = {"name": "ThinObject5k-TR",
                 "im_dir": "./data/thin_object_detection/ThinObject5K/images_train",
                 "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_train",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_fss = {"name": "FSS",
                 "im_dir": "./data/cascade_psp/fss_all",
                 "gt_dir": "./data/cascade_psp/fss_all",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_duts = {"name": "DUTS-TR",
                 "im_dir": "./data/cascade_psp/DUTS-TR",
                 "gt_dir": "./data/cascade_psp/DUTS-TR",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_duts_te = {"name": "DUTS-TE",
                 "im_dir": "./data/cascade_psp/DUTS-TE",
                 "gt_dir": "./data/cascade_psp/DUTS-TE",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_ecssd = {"name": "ECSSD",
                 "im_dir": "./data/cascade_psp/ecssd",
                 "gt_dir": "./data/cascade_psp/ecssd",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_msra = {"name": "MSRA10K",
                 "im_dir": "./data/cascade_psp/MSRA_10K",
                 "gt_dir": "./data/cascade_psp/MSRA_10K",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    # valid set
    dataset_coift_val = {"name": "COIFT",
                 "im_dir": "./data/thin_object_detection/COIFT/images",
                 "gt_dir": "./data/thin_object_detection/COIFT/masks",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_hrsod_val = {"name": "HRSOD",
                 "im_dir": "./data/thin_object_detection/HRSOD/images",
                 "gt_dir": "./data/thin_object_detection/HRSOD/masks_max255",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_thin_val = {"name": "ThinObject5k-TE",
                 "im_dir": "./data/thin_object_detection/ThinObject5K/images_test",
                 "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_test",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_dis_val = {"name": "DIS5K-VD",
                 "im_dir": "./data/DIS5K/DIS-VD/im",
                 "gt_dir": "./data/DIS5K/DIS-VD/gt",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}
    
    dataset_sunrgbd_train = {"name": "SUNRGBD",
                 "im_dir": "data/sunrgbd_meta/image/train",
                 "depth_dir": "data/sunrgbd_meta/sunrgbd_train_depth",
                 "gt_dir": "data/sunrgbd_meta/label37_train",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}

    dataset_sunrgbd_val = {"name": "SUNRGBD_VAL",
                 "im_dir": "data/sunrgbd_meta/image/test",
                 "depth_dir": "data/sunrgbd_meta/sunrgbd_test_depth",
                 "gt_dir": "data/sunrgbd_meta/label37_test",
                 "im_ext": ".jpg",
                 "gt_ext": ".png"}
    
    dataset_nyudv2_train = {"name": "NYUDV2",
                 "im_dir": "data/NYUDepthv2train/RGB",
                 "depth_dir": "data/NYUDepthv2train/Depth",
                 "gt_dir": "data/NYUDepthv2train/Label",
                 "im_ext": ".jpg",
                 "depth_ext": ".npy",
                 "gt_ext": ".png"}

    dataset_nyudv2_val = {"name": "NYUDV2_VAL",
                 "im_dir": "data/NYUDepthv2test/RGB",
                 "depth_dir": "data/NYUDepthv2test/Depth",
                 "gt_dir": "data/NYUDepthv2test/Label",
                 "im_ext": ".jpg",
                 "depth_ext": ".npy",
                 "gt_ext": ".png"}
    

    dataset_scannet_train = {"name": "SCANNET",
                 "im_dir": "data/scannet_frames_25k_train/color",
                 "depth_dir": "data/scannet_frames_25k_train/depth",
                 "gt_dir": "data/scannet_frames_25k_train/instance",
                 "im_ext": ".jpg",
                 "depth_ext": ".png",
                 "gt_ext": ".png"}
    
    dataset_scannet_sem_train = {"name": "SCANNET_SEM",
                 "im_dir": "data/scannet_frames_25k_train/color",
                 "depth_dir": "data/scannet_frames_25k_train/depth",
                 "gt_dir": "data/scannet_frames_25k_train/label",
                 "im_ext": ".jpg",
                 "depth_ext": ".png",
                 "gt_ext": ".png"}
    

    dataset_kitti360_train = {"name": "KITTI360",
                 "im_dir": "data/KITTI-360-train/data_rect",
                 "depth_dir": "data/KITTI-360-train/velodyne_points",
                 "gt_dir": "data/KITTI-360-train/2d_instance",
                 "im_ext": ".png",
                 "depth_ext": ".png",
                 "gt_ext": ".png"}
    
    dataset_deliver_train = {"name": "DELIVER_LIDAR",
                 "im_dir": "data/deliver/img/train",
                 "depth_dir": "data/deliver/lidar/train",
                 "gt_dir": "data/deliver/semantic/train",
                 "im_ext": "_rgb_front.png",
                 "depth_ext": "_lidar_front.png",
                 "gt_ext": "_semantic_front.png"}
    
    dataset_deliver_val = {"name": "DELIVER_LIDAR_VAL",
                 "im_dir": "data/deliver/img/val",
                 "depth_dir": "data/deliver/lidar/val",
                 "gt_dir": "data/deliver/semantic/val",
                 "im_ext": "_rgb_front.png",
                 "depth_ext": "_lidar_front.png",
                 "gt_ext": "_semantic_front.png"}
    
    dataset_deliver_event_train = {"name": "DELIVER_EVENT",
                 "im_dir": "data/deliver/img/train",
                 "depth_dir": "data/deliver/event/train",
                 "gt_dir": "data/deliver/semantic/train",
                 "im_ext": "_rgb_front.png",
                 "depth_ext": "_event_front.png",
                 "gt_ext": "_semantic_front.png"}
    
    dataset_deliver_event_val = {"name": "DELIVER_EVENT_VAL",
                 "im_dir": "data/deliver/img/val",
                 "depth_dir": "data/deliver/event/val",
                 "gt_dir": "data/deliver/semantic/val",
                 "im_ext": "_rgb_front.png",
                 "depth_ext": "_event_front.png",
                 "gt_ext": "_semantic_front.png"}
    

    dataset_event_scape_event_train = {"name": "EVENTSCAPE_EVENT",
                 "im_dir": "data/EventScape_30F/rgb/train",
                 "depth_dir": "data/EventScape_30F/events/train",
                 "gt_dir": "data/EventScape_30F/semantic/train",
                 "im_ext": ".png",
                 "depth_ext": ".png",
                 "gt_ext": ".png"}
    
    dataset_event_scape_event_val = {"name": "EVENTSCAPE_EVENT_VAL",
                 "im_dir": "data/EventScape_30F/rgb/val",
                 "depth_dir": "data/EventScape_30F/events/val",
                 "gt_dir": "data/EventScape_30F/semantic/val",
                 "im_ext": ".png",
                 "depth_ext": ".png",
                 "gt_ext": ".png"}
    
    dataset_dsec_event_val_11c = {"name": "DSEC_EVENT_VAL_11CLASSES",
                 "im_dir": "data/DSEC_30F/images/test",
                 "depth_dir": "data/DSEC_30F/events/test",
                 "gt_dir": "data/DSEC_30F/semantic/test11classes",
                 "im_ext": ".png",
                 "depth_ext": ".png",
                 "gt_ext": ".png"}
    
    dataset_dsec_event_val_19c = {"name": "DSEC_EVENT_VAL_19CLASSES",
                 "im_dir": "data/DSEC_30F/images/test",
                 "depth_dir": "data/DSEC_30F/events/test",
                 "gt_dir": "data/DSEC_30F/semantic/test19classes",
                 "im_ext": ".png",
                 "depth_ext": ".png",
                 "gt_ext": ".png"}

    
    def dataset_stanford_train(area):

        # data/Stanford2D3D/area_1/data/rgb/camera_0a70cd8d4f2b48239aaa5db59719158a_office_12_frame_0_domain_rgb.png
        # data/Stanford2D3D/area_1/data/depth/camera_0a70cd8d4f2b48239aaa5db59719158a_office_12_frame_0_domain_depth.png
        # data/Stanford2D3D/area_1/data/semantic/camera_0a70cd8d4f2b48239aaa5db59719158a_office_12_frame_0_domain_semantic.png
        return {"name": f"Stanford2D3D_{area}",
                 "im_dir": f"data/Stanford2D3D/area_{area}/data/rgb",
                 "depth_dir": f"data/Stanford2D3D/area_{area}/data/depth",
                 "gt_dir": f"data/Stanford2D3D/area_{area}/data/semantic",
                 "im_ext": "domain_rgb.png",
                 "depth_ext": "domain_depth.png",
                 "gt_ext": "domain_semantic.png"}
    def dataset_stanford_train_sem(area):

        # data/Stanford2D3D/area_1/data/rgb/camera_0a70cd8d4f2b48239aaa5db59719158a_office_12_frame_0_domain_rgb.png
        # data/Stanford2D3D/area_1/data/depth/camera_0a70cd8d4f2b48239aaa5db59719158a_office_12_frame_0_domain_depth.png
        # data/Stanford2D3D/area_1/data/semantic/camera_0a70cd8d4f2b48239aaa5db59719158a_office_12_frame_0_domain_semantic.png
        return {"name": f"Stanford2D3D_{area}_SEM",
                 "im_dir": f"data/Stanford2D3D/area_{area}/data/rgb",
                 "depth_dir": f"data/Stanford2D3D/area_{area}/data/depth",
                 "gt_dir": f"data/Stanford2D3D/area_{area}/data/semantic_sem",
                 "im_ext": "domain_rgb.png",
                 "depth_ext": "domain_depth.png",
                 "gt_ext": "domain_semantic.png"}


    # train_datasets = [dataset_dis, dataset_thin, dataset_fss, dataset_duts, dataset_duts_te, dataset_ecssd, dataset_msra]
    # valid_datasets = [dataset_dis_val, dataset_coift_val, dataset_hrsod_val, dataset_thin_val] 
    # train_datasets = [dataset_sunrgbd_train]
    # valid_datasets = [dataset_sunrgbd_val] 
    
    train_datasets = [dataset_scannet_sem_train] + [dataset_stanford_train_sem(i) for i in [1,2,3,4,6] ]
    valid_datasets = [dataset_nyudv2_val] 


    # train_datasets = [dataset_event_scape_event_train, dataset_deliver_event_train]
    # valid_datasets = [dataset_event_scape_event_val] 
    args = get_args_parser()
    # net = MaskDecoderHQ(args.model_type) 

    main(None, train_datasets, valid_datasets, args)
