# Author: Siddharth Mittal

import argparse
import torch

from tensorboardX import SummaryWriter
from i3dmod import modI3D
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", help="number of iterations to run on", type=int, default=10000)
parser.add_argument("--gpus", help="Number of GPUs to run on", type=int, default=1)
parser.add_argument("--lr", help="starting lr", type=float, default=1e-2)
parser.add_argument("--numw", help="number of workers on loading data", type=int, default=4)
parser.add_argument("--batch", help="batch-size", type=int, default=6)
parser.add_argument("--testbatch", help="test batch-size", type=int, default=6)
parser.add_argument("--trainlist", help="Training file list", type=str, default='../list/trainlist01.txt')
parser.add_argument("--testlist", help="Testing file list", type=str, default='../list/protestlist01.txt')
parser.add_argument('--modality', type=str, default='rgb', help='RGB vs Flow')
parser.add_argument('--ft', type=str, default=False, help='Finetune the model or not')
parser.add_argument("--eval", help="Just put to keep constistency with original model (rgb)", type=str, default='rgb')

# ../model/model_rgb.pth
args = parser.parse_args()

_SAMPLE_VIDEO_FRAMES = 64
_IMAGE_SIZE = 224
_NUM_CLASSES = 101

_BATCH_SIZE = args.batch
_TEST_BATCH_SIZE = args.testbatch
_NUM_GPUS = args.gpus
_EPOCHS = args.epochs
_LEARNING_RATE = args.lr
_TRAIN_LIST = args.trainlist
_TEST_LIST = args.testlist
_TEST_FREQ = 20
_NUM_W = args.numw
_MODALITY = args.modality
_FT = args.ft
eval_type = args.eval

_LOGDIR = '../logs/' + str(_LEARNING_RATE) + '_' + str(_EPOCHS) + '_' + _TRAIN_LIST.split('/')[2].split('.')[0]


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

    train_dataset = TSNDataSet("", _TRAIN_LIST, num_segments=1, new_length=64, modality=_MODALITY, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_BATCH_SIZE, shuffle=True, num_workers=_NUM_W)

    test_dataset = TSNDataSet("", _TEST_LIST, num_segments=1, new_length=64, modality=_MODALITY, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_TEST_BATCH_SIZE, shuffle=True, num_workers=_NUM_W)

    return train_loader, test_loader

def run(model, train_loader, criterion, optimizer, train_writer, scheduler, test_loader=None, test_writer=None):
    # Load data

    avg_loss = AverageMeter()

    global_step = 1
    train_points = len(train_loader)
    for j in range(_EPOCHS):

        print("Epoch Number: %d" % (j + 1))
        get_test_accuracy(model, test_loader)
        scheduler.step()
        for i, (input_3d, target) in enumerate(train_loader):

            # print("Input shape: ",input_3d.size())
            # print("Target shape: ",target.size())

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
            optimizer.step()
            optimizer.zero_grad()

            if global_step % int(train_points/6) == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad != None:
                        # print("Histogram for[Name]: ",name)
                        train_writer.add_histogram(name, param.clone().cpu().data.numpy(),global_step)
                        train_writer.add_histogram(name + '/gradient', param.grad.clone().cpu().data.numpy(),global_step)

            if test_loader is not None and global_step % int(train_points/2) == 0:
                get_test_accuracy(model, test_loader)


            global_step += 1

        # scheduler.step(avg_loss.avg)
    get_test_accuracy(model, test_loader)
    train_writer.close()


def main():
    train_loader, test_loader = get_set_loader()

    # args order: modality, num_c, finetune, dropout
    model = modI3D(modality=_MODALITY, finetune = _FT)
    if _NUM_GPUS > 1:
        model = torch.nn.DataParallel(model)

    print_learnables(model)
    model.train()
    model.cuda()

    # with cross entropy loss we don't require to compute softmax, nor do we need one-hot encodings
    loss = torch.nn.CrossEntropyLoss()
    sgd = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=_LEARNING_RATE, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(sgd,'min',patience=2,verbose=True,threshold=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(sgd,milestones=[2,5])
    writer = SummaryWriter(_LOGDIR)
    run(model, train_loader, loss, sgd, writer, scheduler, test_loader=test_loader)
    print("Logged in: ",_LOGDIR)
    # run(model, train_loader, loss, sgd, writer)


if __name__ == '__main__':
    main()
