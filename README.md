I3D Models for UCF-101
=======================

- The main script is ucf_pt.py

## Options

| Option | Default | Meaning |
|--------|---------|---------|
| --epochs | 50 | . |
| --gpus | 1 | . |
| --lr | 0.01 | starting lr |
| --numw | 8 | . |
| --batch | 6 | . |
| --testbatch | 6 | Test batch size |
| --trainlist | ../list/trainlist01.txt | trainlist |
| --testlist | ../list/protestlist01.txt | testlist |
| --modality | rgb | rgb / flow / rgbdsc / flowdsc / flyflow |
| --wts | rgb | (rgb/flow) which weights to load | 
| --random | False | To let the first layer have random weights |
| --ft | False | Finetune or not |
| --mean| False | Use mean and unsquezzing or linear transformation |
| --sched | False | Use a scheduler or not |
| --nstr | None | name string(used to add to default name) |
| --dog | False | Use of Difference of Gaussians(not trainable) as the first filter |

## Modality details

- rgb:
Simply use the rgb data in /mnt/data1/UCF-101

- rgbdsc:
Use the rgb data in /mnt/data1/UCF-101, convert to grayscale, and then send to Reichardt DS8

- flow:
Simply use the flow data in /mnt/data1/UCF-101_old

- flowdsc
Use the flow data in /mnt/data1/UCF-101_old, but transform the data from 2 channels to 8 using the transformation matrix.
[Not thresholding the output currently]

- flyflow
Use only 2 channel Reichardt output for rgb data in /mnt/data1/UCF-101

## Example

```
CUDA_VISIBLE_DEVICES='4,5' python ucf_pt.py --lr 0.01 --epochs 6 --trainlist ../list/trainlist01.txt --testlist ../list/protestlist01.txt
```

## [Results](https://docs.google.com/spreadsheets/d/1S2Qb1E3a6jRwpmNY7_TuHdeCRzgPuqV4pldVhNZSK6k/edit?usp=sharing)

## Reference

- Yana Hasson
- Paper [*Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset*](https://arxiv.org/abs/1705.07750) by Joao Carreira and Andrew Zisserman.
