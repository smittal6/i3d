# Author: Siddharth Mittal
import os
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

from i3dpt import I3D
from i3dpt import Unit3Dpy
from dataset import TSNDataSet
from tensorboardX import SummaryWriter
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", help="number of iterations to run on", type=int, default=10000)
parser.add_argument("--gpus", help="Number of GPUs to run on", type=int, default=1)
parser.add_argument("--lr", help="starting lr", type=float, default=1e-2)
parser.add_argument("--batch", help="batch-size", type=int, default=6)
parser.add_argument("--eval", help="Just put to keep constistency with original model (rgb)", type=str, default='rgb')
parser.add_argument("--trainlist", help="Training file list", type=str, default='../list/trainlist01.txt')
parser.add_argument("--testlist", help="Testing file list", type=str, default='../list/protestlist01.txt')
parser.add_argument('--rgb_weights_path', type=str, default='../model/model_rgb.pth', help='Path to model state_dict')
args = parser.parse_args()

_SAMPLE_VIDEO_FRAMES = 64
_IMAGE_SIZE = 224
_NUM_CLASSES = 101

_BATCH_SIZE = args.batch
_NUM_GPUS = args.gpus
_EPOCHS = args.epochs
_LEARNING_RATE = args.lr
_TRAIN_LIST = args.trainlist
_TEST_LIST = args.testlist
_TEST_FREQ = 20
eval_type = args.eval

_LOGDIR = '../ftlogs/' + str(_LEARNING_RATE) + '_' + str(_EPOCHS) + '_' + _TRAIN_LIST.split('/')[2].split('.')[0]

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

class mod_I3D(torch.nn.Module):

    def __init__(self, num_c=101, dropout_prob=0.5):

        super(mod_I3D, self).__init__()

        # Initialize pytorch I3D
        self.i3d = I3D(num_classes=400, dropout_prob=0.5)
        # Can't change the classes because weight loading will cause issues.

        self.num_c = num_c

        self.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.conv3d_0c_1x1 = Unit3Dpy(
            in_channels=1024,
            out_channels=self.num_c,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)
        self.softmax = torch.nn.Softmax(1)

        self.i3d.eval()
        self.i3d.load_state_dict(torch.load(args.rgb_weights_path))
        print("Pretrained i3D weights restored")
        # i3nception_pt.train()

        self.i3d.cuda()
        print("Pretrained i3D shifted to CUDA")

        # for name, param in self.i3d.named_parameters():
            # print("Setting requires_grad as False for: ", name)
            # param.requires_grad = False

    def forward(self, inp):
        # Loaded i3D section
        out = self.i3d.conv3d_1a_7x7(inp)
        out = self.i3d.maxPool3d_2a_3x3(out)
        out = self.i3d.conv3d_2b_1x1(out)
        out = self.i3d.conv3d_2c_3x3(out)
        out = self.i3d.maxPool3d_3a_3x3(out)
        out = self.i3d.mixed_3b(out)
        out = self.i3d.mixed_3c(out)
        out = self.i3d.maxPool3d_4a_3x3(out)
        out = self.i3d.mixed_4b(out)
        out = self.i3d.mixed_4c(out)
        out = self.i3d.mixed_4d(out)
        out = self.i3d.mixed_4e(out)
        out = self.i3d.mixed_4f(out)
        out = self.i3d.maxPool3d_5a_2x2(out)
        out = self.i3d.mixed_5b(out)
        out = self.i3d.mixed_5c(out)

        out = self.avg_pool(out)
        out = self.dropout(out)
        out = self.conv3d_0c_1x1(out)
        out = out.squeeze(3)
        out = out.squeeze(3)
        out = out.mean(2)
        out_logits = out
        out = self.softmax(out_logits)
        return out, out_logits


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


def get_set_loader():
    i_h = 240
    i_w = 320

    # Take the min value, and use it for the ratio
    min_ = min(i_w, i_h)
    ratio = float(min_) / 256

    new_width = int(float(i_w) / ratio)
    new_height = int(float(i_h) / ratio)

    train_transform = transforms.Compose(
        [transforms.Resize([new_height, new_width]), transforms.RandomCrop(size=_IMAGE_SIZE), transforms.ToTensor(), PixRescaler()])

    test_transform = transforms.Compose(
        [transforms.Resize([new_height, new_width]), transforms.CenterCrop(size=_IMAGE_SIZE), transforms.ToTensor(), PixRescaler()])

    train_dataset = TSNDataSet("", _TRAIN_LIST, num_segments=1, new_length=64, modality='RGB', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_BATCH_SIZE, shuffle=True, num_workers=8)

    test_dataset = TSNDataSet("", _TEST_LIST, num_segments=1, new_length=64, modality='RGB', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_BATCH_SIZE, shuffle=True, num_workers=8)

    return train_loader, test_loader


def get_test_accuracy(model,test_loader):

    # print("Obtaining the test accuracy now")
    
    model.eval()
    accs = []
    for i, (test, labels) in enumerate(tqdm(test_loader)):
        # print("test batch number: ",i)
        test = torch.autograd.Variable(test.cuda(), volatile=True)
        labels = torch.autograd.Variable(labels.cuda())
        _, output = model(test)
        accs.append(accuracy(output, labels))

    print("--------\t\t\t\t Test Accuracy: %0.5f " % np.mean(accs))
    model.train()


def run(model, train_loader, criterion, optimizer, train_writer, scheduler, test_loader=None, test_writer=None):
    # Load data

    avg_loss = AverageMeter()

    global_step = 1
    train_points = len(train_loader)
    for j in range(_EPOCHS):

        print("Epoch Number: %d" % (j + 1))
        # get_test_accuracy(model, test_loader)

        if scheduler is not None:
            scheduler.step()
        for i, (input_3d, target) in enumerate(train_loader):

            # print("Input shape: ",input_3d.size())
            # print("Target shape: ",target.size())

            optimizer.zero_grad()
            # print("Targets: ",target)

            # Prepare data for pytorch forward pass
            input_3d_var = torch.autograd.Variable(input_3d.cuda())
            target = torch.autograd.Variable(target.cuda())

            # Pytorch forward pass
            out_pt, logits = model(input_3d_var)
            loss = criterion(logits, target)
            avg_loss.update(loss)

            print("Epoch: %d, Step: [%d / %d], Training Loss: %0.5f, Average: %0.5f" % (j+1, i+1, train_points, loss, avg_loss.avg))

            train_writer.add_scalar('Loss', loss, global_step)
            train_writer.add_scalar('Avg Loss', avg_loss.avg, global_step)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # print("Histogram for[Name]: ",name)
                    train_writer.add_histogram(name, param.clone().cpu().data.numpy(),global_step)
                    train_writer.add_histogram(name + '/gradient', param.grad.clone().cpu().data.numpy(),global_step)

            if test_loader is not None and global_step % int(train_points/2) == 0:
                get_test_accuracy(model, test_loader)

            optimizer.step()

            global_step += 1

        # scheduler.step(avg_loss.avg)
    get_test_accuracy(model, test_loader)
    train_writer.close()


def print_learnables(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Learning for: ", name)


def main():
    train_loader, test_loader = get_set_loader()

    model = mod_I3D()
    if _NUM_GPUS > 1:
        print("Using %d GPUs"%_NUM_GPUS)
        model = torch.nn.DataParallel(model)

    print_learnables(model)
    model.train()
    model.cuda()

    # with cross entropy loss we don't require to compute softmax, nor do we need one-hot encodings
    loss = torch.nn.CrossEntropyLoss()
    sgd = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=_LEARNING_RATE, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(sgd,'min',patience=2,verbose=True,threshold=0.0001)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(sgd,milestones=[2,5])
    scheduler = None
    writer = SummaryWriter(_LOGDIR)
    run(model, train_loader, loss, sgd, writer, scheduler, test_loader=test_loader)
    print("Logged in: ",_LOGDIR)
    # run(model, train_loader, loss, sgd, writer)


if __name__ == '__main__':
    main()
