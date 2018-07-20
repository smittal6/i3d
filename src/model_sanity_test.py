from utils import *
from i3dmod import *
import torch
from torch.autograd import Variable

'''
This file tests the model's forward pass for differnt modalities
'''

# Modality-Weights


# Combination 1: rgb-rgb
def comb1():
    mod1 = modI3D('rgb','rgb',False,True)
    mod2 = smallI3D('rgb','rgb',False,True)

    print("Shifting to CPUs")
    mod1.cpu()
    mod2.cpu()

    sample = Variable(torch.rand(1,3,64,224,224))

    _, l1 = mod1(sample)
    _, l2 = mod2(sample)
    del mod1
    del mod2

# Combination 2: flow-flow
def comb2():
    mod1 = modI3D('flow','flow',False,True)
    mod2 = smallI3D('flow','flow',False,True)

    print("Shifting to CPUs")
    mod1.cpu()
    mod2.cpu()

    sample = Variable(torch.rand(1,2,64,224,224))

    _, l1 = mod1(sample)
    _, l2 = mod2(sample)
    del mod1
    del mod2


# Combination 3: flyflow-flow
def comb3():
    mod1 = modI3D('flyflow','flow',False,True)
    mod2 = smallI3D('flyflow','flow',False,True)

    print("Shifting to CPUs")
    mod1.cpu()
    mod2.cpu()

    sample = Variable(torch.rand(1,8,64,224,224))

    _, l1 = mod1(sample)
    _, l2 = mod2(sample)
    del mod1
    del mod2

# Combination 4: rgbdsc-flow
def comb4():
    mod1 = modI3D('rgbdsc','flow',False,True)
    mod2 = smallI3D('rgbdsc','flow',False,True)

    print("Shifting to CPUs")
    mod1.cpu()
    mod2.cpu()

    sample = Variable(torch.rand(1,8,64,224,224))

    _, l1 = mod1(sample)
    _, l2 = mod2(sample)
    del mod1
    del mod2

def main():
    comb1()
    comb2()
    comb3()
    comb4()

if __name__ == '__main__':
        main()

