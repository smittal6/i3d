from dataset import TSNDataSet
from tqdm import tqdm
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class PixRescaler(object):

    def __init__(self, high=1, low=-1):
        self.high = high
        self.low = low

    def __call__(self, tensor):
        '''
            Currently this is very specific to DeepMind's UCF101 training case
        '''
        return 2*tensor-1


def print_learnables(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Learning for: ", name)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_test_accuracy(model, test_loader):
    # print("Obtaining the test accuracy now")

    model.eval()
    accs = []
    for i, (test, labels) in enumerate(tqdm(test_loader)):
        # print("test batch number: ",i)
        test = torch.autograd.Variable(test.cuda(), volatile=True)
        labels = torch.autograd.Variable(labels.cuda())
        _, output = model(test)
        accs.append(accuracy(output, labels))

    print("--------\t\t\t Test Accuracy: %0.5f " % np.mean(accs))
    model.train()
    return np.mean(accs)

def save_checkpoint(args, state, is_best, filename='checkpoint_4B.pth.tar'):
    filename = '_'.join((args.modality.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.modality.lower(), 'model_best_4B.pth.tar'))
        shutil.copyfile(filename, best_name)
