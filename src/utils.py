from tqdm import tqdm
import numpy as np
import torch
from torch._six import string_classes, int_classes
import collections
import shutil
import scipy.signal

from itertools import chain

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


def rescale(matrix, scale_min=0, scale_max=255):
    """
    Rescale matrix in a given range, element wise
    """
    matrix = matrix.astype(np.float32)
    matrix = (matrix - matrix.min()) * ((scale_max - scale_min) / np.ptp(matrix)) + scale_min
    return matrix


def print_learnables(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Learning for: %s, Params: %d"%(name,np.prod(param.size())))


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


def get_test_accuracy(model, test_loader, model_stream2=None):
    # print("Obtaining the test accuracy now")

    model.eval()
    accs = []
    if model_stream2 is None:
        for i, (test, labels) in enumerate(tqdm(test_loader)):
            # print("test batch number: ",i)
            test = torch.autograd.Variable(test.cuda(), volatile=True)
            labels = torch.autograd.Variable(labels.cuda())
            _, output = model(test)
            accs.append(accuracy(output, labels))

    else:
        model_stream2.eval()
        for i, (test1, test2, labels) in enumerate(tqdm(test_loader)):
            # print("test batch number: ",i)
            test1 = torch.autograd.Variable(test1.cuda(), volatile=True)
            test2 = torch.autograd.Variable(test2.cuda(), volatile=True)
            labels = torch.autograd.Variable(labels.cuda())
            _, output1 = model(test1)
            _, output2 = model_stream2(test2)
            accs.append(accuracy(output1 + output2, labels))

        model_stream2.train()

    print("--------\t\t\tCurrent Test Accuracy: %0.5f " % np.mean(accs))
    model.train()
    return np.mean(accs)


def save_checkpoint(args, state, is_best, filename='checkpoint_4B.pth.tar'):
    filename = '_'.join((args.modality.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.modality.lower(), 'model_best_4B.pth.tar'))
        shutil.copyfile(filename, best_name)

def interruptHandler(args, model, writer, test_loader, best_prec1, global_step, j, model_stream2=None):

    # find the accuracy before shutting
    acc = get_test_accuracy(model, test_loader, model_stream2)

    # model saving 
    if acc.data[0] > best_prec1:
        print("Saving this model as the best.")
        best_prec1 = acc.data[0]
        print("Best Accuracy till now: %0.5f " % best_prec1)
        save_checkpoint(args, {'epoch': j + 1,'state_dict': model.state_dict(),'best_prec1': best_prec1}, True)

    # store the grads
    print("Storing the gradients for Tensorboard")

    if model_stream2 is None:
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # print("Histogram for[Name]: ",name)
                writer.add_histogram(name, param.clone().cpu().data.numpy(),global_step+1)
                writer.add_histogram(name + '/gradient', param.grad.clone().cpu().data.numpy(),global_step+1)
    else:
        for name, param in chain(model.named_parameters(),model_stream2.named_parameters()):
            if param.requires_grad and param.grad is not None:
                # print("Histogram for[Name]: ",name)
                writer.add_histogram(name, param.clone().cpu().data.numpy(),global_step+1)
                writer.add_histogram(name + '/gradient', param.grad.clone().cpu().data.numpy(),global_step+1)


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    _use_shared_memory = False
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    raise TypeError((error_msg.format(type(batch[0]))))


def my_collate(batch):
    '''
    Filter out the None elements [introduced as a part to ignore the missing flow files]
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def gkern(kernlen=7, std=3, max =1):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = scipy.signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d * max
    
    
def kernels(size=5):
    m = gkern(kernlen=size, std = 5, max=0.5)
    m1 = gkern(kernlen=size, std = 1, max=2)
    x = (m1 - m)
    x = x.reshape(1,size,size)
    x = x/np.max(x)
    print("The filter: ")
    print(x)
    # print("The sum of the kernel: ",np.sum(x))
    return x
