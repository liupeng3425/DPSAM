# source ~/anaconda3/bin/activate ovseg

# --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
# --checkpoint ./pretrained_checkpoint/sam_vit_h_4b8939.pth \
# --checkpoint ./pretrained_checkpoint/sam_vit_l_0b3195.pth \

mtype="vit_l_adapter_vit"
# opath="work_dirs/hq_sam_l_adapter_rn_enc"
opath="work_dirs/hq_sam_${mtype}_sem"
checkpoint="./pretrained_checkpoint/sam_vit_l_0b3195.pth"
# CUDA_VISIBLE_DEVICES=2,4,5,7 \
python -m torch.distributed.launch --nproc_per_node=8 --master_port 16439 train_decoder_sem.py \
    --checkpoint ${checkpoint} \
    --model-type ${mtype} \
    --output ${opath} \
    --max_epoch_num 15 \
    --batch_size_train 4 \
    --batch_size_valid 4 \
    --learning_rate 0.00001 \
    --find_unused_params

