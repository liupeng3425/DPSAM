# Segmenting Anything in the Dark via Depth Perception
[TMM] Official implementation of Segmenting Anything in the Dark via Depth Perception.

## Training
The conda environment setup can follow the instructions in [HQ-SAM](https://github.com/SysCV/sam-hq).

Download the SAM weights provided by [SAM](https://github.com/facebookresearch/segment-anything) and put them in the folder `./train/pretrained_checkpoint/`.

A training job can be launched using:

```shell
cd train
bash script_train_adapter_vit_enc_sem.sh
```

Citation
---------------
If you find this repo useful in your research or refer to the provided baseline results, please star :star: this repository and consider citing :pencil::
```
@article{liu2025segmenting,
  title={Segmenting anything in the dark via depth perception},
  author={Liu, Peng and Deng, Jinhong and Duan, Lixin and Li, Wen and Lv, Fengmao},
  journal={IEEE Transactions on Multimedia},
  year={2025},
  publisher={IEEE}
}
```

## Acknowledgments
- Thanks [HQ-SAM](https://github.com/SysCV/sam-hq) and [SAM](https://github.com/facebookresearch/segment-anything) for their public code and released models.
