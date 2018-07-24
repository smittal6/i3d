I3D Models for UCF-101
=======================

- The main script is ucf_pt.py
- This script uses the pretrained weights for i3d: converted from TF to PyTorch [courtesy Yana Hasson]
- Logdir naming convention:
    ```
    logs/_MODALITY/_WTS _ _LEARNING_RATE _ EPOCHS
    ```
## Dependencies

- pytorch=0.3
- tensorboardX
- tqdm
- torchvision

## Options

| Option | Default | Meaning |
|--------|---------|---------|
| --trainlist | ../list/flowtrainlist01.txt | trainlist |
| --testlist | ../list/flowtestlist01.txt | testlist |
| --nstr | None | name string(used to add to default name) |
| --modality | rgb | rgb / flow / rgbdsc / flowdsc / flyflow |
| --modality2 | None | Second stream mode: rgb / flow / rgbdsc / flowdsc / flyflow / edr1|
| --wts | rgb | (rgb/flow) which weights to load | 
| --mean| False | Use mean and unsquezzing or linear transformation |
| --random | False | To let the first layer have random weights |
| --dog | False | Use of Difference of Gaussians(not trainable) as the first filter |
| --rdirs | [0,1,2,3,4,5,6,7] | Reichardt directions to extract |
| --epochs | 40 | . |
| --lr | 0.001 | starting lr |
| --batch | 6 | . |
| --testbatch | 6 | Test batch size |
| --sched | False | Use a scheduler or not |
| --lr_steps | [2,6,10,13] | scheduler steps |
| --thres | None | Threshold the input to the network to get sparsity |
| --gpus | 1 | . |
| --numw | 8 | . |
| --resume | None | Resume training or not |
| --ft | True | Finetune or not |

## Modality details

- rgb:
Simply use the rgb data in /mnt/data1/UCF-101

- rgbdsc:
Use the rgb data in /mnt/data1/UCF-101, convert to grayscale, and then send to Reichardt DS8

- flow:
Simply use the flow data in /mnt/data1/UCF-101_old

- flowdsc:
Use the flow data in /mnt/data1/UCF-101_old, but transform the data from 2 channels to 8 using the transformation matrix.
[Not thresholding the output currently]

- flyflow:
    - Use only 2 channel Reichardt output for rgb data in /mnt/data1/UCF-101
    - With flyflow and specifying more than 2 rdirs, by default mean will become active to ensure model integrity. That is for the first layer of convolutions, weights will be averaged, and then finetuned.
    - rdirs convention: 
        * 0,1: vertical(1), vertical(-1)
        * 2,3: diagnol1(2), diagnol1(-2)
        * 4,5: horizontal(3), horizontal(-3)
        * 6,7: diagnol2(4), diagnol2(-4)
    - This is just a variant of rgbdsc with option of chosing directions


## List details

- Training list: flowtrainlist01.txt
- Testing list: flowtestlist01.txt
- The data folder for each video has flow_x and flow_y along with img for all the timesteps in the video.


## Example

- Running flyflow modality, with flow weights and 2 GPUs and no scheduler
```
CUDA_VISIBLE_DEVICES="4,5" python ucf_pt.py --modality flyflow --wts flow --epochs 40 --ft True --lr 0.001 --gpus 2 --trainlist ../list/flowtrainlist01.txt --testlist ../list/flowtestlist01.txt
```

- Running flowdsc modality, with flow weights and 2 GPUs and no scheduler
```
CUDA_VISIBLE_DEVICES="4,5" python ucf_pt.py --modality flyflow --wts flow --epochs 40 --ft True --lr 0.001 --gpus 2 --trainlist ../list/flowtrainlist01.txt --testlist ../list/flowtestlist01.txt
```

- Running rgb modality with difference of gaussian, with flow weights and 2 GPUs and no scheduler
```
CUDA_VISIBLE_DEVICES="2,7" python ucf_pt.py --modality rgb --wts rgb --random True --dog True --nstr frand_nsched_dog
```

## Steps to add modality

1. In get_set_loader() funtion found in ucf_pt.py modify modlist to include new modality name
2. In file dataset.py:
    - Change *_load_image()* function to include your new modality
    - If required, import the desired modality function, and include it in the *get()* method.
    - Before the if-elifs: the shape of process_data is ~ [Channels, T, H, W]
3. In transforms.py:
    - Modify Stack function to include the details for new modality

## [Results](https://docs.google.com/spreadsheets/d/1S2Qb1E3a6jRwpmNY7_TuHdeCRzgPuqV4pldVhNZSK6k/edit?usp=sharing)

## Reference

- [TSN Codebase](https://github.com/yjxiong/tsn-pytorch)
- [Yana Hasson](https://github.com/hassony2)
- Paper [*Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset*](https://arxiv.org/abs/1705.07750) by Joao Carreira and Andrew Zisserman.
