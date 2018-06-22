I3D Models for UCF-101
=======================

- The main script is ucf_pt.py

## Options

| Option | Default | Meaning |
|--------|---------|---------|
| --epochs | 100 | . |
| --gpus | 1 | . |
| --lr | 0.01 | starting lr |
| --numw | 8 | . |
| --batch | 6 | . |
| --testbatch | 6 | Test batch size |
| --trainlist | ../list/trainlist01.txt | trainlist |
| --testlist | ../list/protestlist01.txt | testlist |
| --modality | rgb | rgb or flow or rgbdsc or flowdsc |
| --wts | rgb | which weights to load | 
| --ft | False | Finetune or not |
| --sched | False | Use a scheduler or not |

```
CUDA_VISIBLE_DEVICES='4' python ucf_pt.py --lr 0.01 --epochs 6 --trainlist ../list/trainlist01.txt --testlist ../list/protestlist01.txt
```

## Reference

- Yana Hasson
- Paper [*Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset*](https://arxiv.org/abs/1705.07750) by Joao Carreira and Andrew Zisserman to PyTorch. 
