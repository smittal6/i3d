# Author: Siddharth Mittal

import argparse
import torch
import time
import sys

from tensorboardX import SummaryWriter
from i3dmod import modI3D
from utils import *
from transforms import *

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", help="number of iterations", type=int, default=100)
parser.add_argument("--gpus", help="Number of GPUs", type=int, default=1)
parser.add_argument("--lr", help="starting lr", type=float, default=1e-2)
parser.add_argument("--numw", help="number of workers on loading data", type=int, default=8)
parser.add_argument("--batch", help="batch-size", type=int, default=6)
parser.add_argument("--testbatch", help="test batch-size", type=int, default=6)
parser.add_argument("--trainlist", help="Training file list", type=str, default='../list/trainlist01.txt')
parser.add_argument("--testlist", help="Testing file list", type=str, default='../list/protestlist01.txt')
parser.add_argument('--modality', type=str, default='rgb', help='rgb / rgbdsc / flow / flowdsc')
parser.add_argument('--wts', type=str, default='rgb', help='rgb/flow')
parser.add_argument('--resume', type=str, default=None, help='Resume training from this file')
parser.add_argument('--ft', type=bool, default=False, help='Finetune the model or not')
parser.add_argument('--sched', type=str, default=False, help='Use a scheduler or not')
parser.add_argument('--mean', type=bool, default=False, help='While transforming the weights use mean or not')
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
_USE_SCHED = args.sched
_TRAIN_LIST = args.trainlist
_TEST_LIST = args.testlist
_TEST_FREQ = 20
_NUM_W = args.numw
_MODALITY = args.modality
_WTS = args.wts
_FT = args.ft
eval_type = args.eval


# dsc1 uses rgb weights with mean, while dsc2 uses flow weights with either mean or transformation
print("Finetune: ",str(_FT))
if _FT:
    _LOGDIR = '../ftlogs/' + _MODALITY + '/' + str(_LEARNING_RATE) + '_' + str(_EPOCHS) + '_' + _TRAIN_LIST.split('/')[2].split('.')[0]
else:
    _LOGDIR = '../logs/' + _MODALITY + '/' + str(_LEARNING_RATE) + '_' + str(_EPOCHS) + '_' + _TRAIN_LIST.split('/')[2].split('.')[0]


def get_set_loader():

    i_h = 240
    i_w = 320

    # Take the min value, and use it for the ratio
    min_ = min(i_w, i_h)
    ratio = float(min_) / 256

    new_width = int(float(i_w) / ratio)
    new_height = int(float(i_h) / ratio)

    # if _MODALITY == 'rgb' or _MODALITY == 'flow':
        # train_transform = transforms.Compose(
            # [transforms.Resize([new_height, new_width]), transforms.RandomCrop(size=_IMAGE_SIZE), transforms.ToTensor(), PixRescaler()])
    # else:
        # train_transform = transforms.Compose(
            # [transforms.Resize([new_height, new_width]), transforms.CenterCrop(size=_IMAGE_SIZE), transforms.ToTensor(), PixRescaler()])
    train_transform = transforms.Compose(
        [GroupScale(size=256),GroupRandomCrop(size=_IMAGE_SIZE), Stack(modality=_MODALITY), ToTorchFormatTensor()])

    # test_transform = transforms.Compose(
        # [transforms.Resize([new_height, new_width]), transforms.CenterCrop(size=_IMAGE_SIZE), transforms.ToTensor(), PixRescaler()])
    test_transform = transforms.Compose(
        [GroupScale(size=256), GroupCenterCrop(size=_IMAGE_SIZE), Stack(modality=_MODALITY), ToTorchFormatTensor()])

    train_dataset = TSNDataSet("", _TRAIN_LIST, num_segments=1, new_length=64, modality=_MODALITY, 
                               image_tmpl="{:05d}.jpg" if _MODALITY in ["rgb", "rgbdsc"] else "flow_"+"{}_{:05d}.jpg",
                               transform=train_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_BATCH_SIZE, shuffle=True, num_workers=_NUM_W)

    test_dataset = TSNDataSet("", _TEST_LIST, num_segments=1, new_length=64, modality=_MODALITY, 
                              image_tmpl="{:05d}.jpg" if _MODALITY in ["rgb", "rgbdsc"] else "flow_"+"{}_{:05d}.jpg",
                              transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_TEST_BATCH_SIZE, shuffle=True, num_workers=_NUM_W)

    return train_loader, test_loader

def run(model, train_loader, criterion, optimizer, train_writer, scheduler, test_loader=None, test_writer=None):

    avg_loss = AverageMeter()
    train_acc = AverageMeter()


    train_points = len(train_loader)
    global j
    global global_step
    global best_prec1
    global_step = 1
    best_prec1 = 0.0
    for j in range(_EPOCHS):

        print("Epoch Number: %d" % (j + 1))
        # get_test_accuracy(model, test_loader)
        if scheduler is not None:
            scheduler.step()
            for param_group in optimizer.param_groups:
                print("Learning Rate: ",param_group['lr'])

        for i, (input_3d, target) in enumerate(train_loader):

            # print("Input shape: ",input_3d.size())
            # print("Target shape: ",target.size())

            # Prepare data for pytorch forward pass
            start = time.time()
            input_3d_var = torch.autograd.Variable(input_3d.cuda())
            target = torch.autograd.Variable(target.cuda())

            # Pytorch forward pass
            out_pt, logits = model(input_3d_var)

            # compute the loss and update the meter
            loss = criterion(logits, target)
            avg_loss.update(loss.data[0])

            # compute the training accuracy
            prec1 = accuracy(logits, target)
            train_acc.update(prec1[0],input_3d.size(0))


            print("Ep: %d, Step: [%d / %d], Loss: %0.5f, Avg: %0.4f, Acc: %0.4f, Time: %0.3f" % (j+1, i+1, train_points, loss.data[0], avg_loss.avg, train_acc.avg, time.time()-start))

            train_writer.add_scalar('Loss', loss, global_step)
            train_writer.add_scalar('Avg Loss', avg_loss.avg, global_step)
            loss.backward()
            optimizer.step()

            if global_step % int(train_points/4) == 0:
                print("Storing the gradients for Tensorboard")
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # print("Histogram for[Name]: ",name)
                        train_writer.add_histogram(name, param.clone().cpu().data.numpy(),global_step)
                        train_writer.add_histogram(name + '/gradient', param.grad.clone().cpu().data.numpy(),global_step)

            if test_loader is not None and global_step % int(train_points/2) == 0:
                acc = get_test_accuracy(model, test_loader)
                # print("best_prec1: ",best_prec1)
                # print("Type of acc.data: ",type(acc.data[0]))
                if acc.data[0] > best_prec1:
                    print("Saving this model as the best.")
                    best_prec1 = acc.data[0]
                    save_checkpoint(args, {'epoch': j + 1,
                                     'state_dict': model.state_dict(),
                                     'best_prec1': best_prec1}, True)
            
            optimizer.zero_grad()
            global_step += 1

        # scheduler.step(avg_loss.avg)
    get_test_accuracy(model, test_loader)
    train_writer.close()


def main():
    global j
    global global_step
    global best_prec1

    train_loader, test_loader = get_set_loader()

    # args order: modality, num_c, finetune, dropout
    model = modI3D(modality=_MODALITY, weights=_WTS, mean=args.mean, finetune = _FT)
    if _NUM_GPUS > 1:
        model = torch.nn.DataParallel(model)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    print_learnables(model)
    model.train()
    model.cuda()

    # with cross entropy loss we don't require to compute softmax, nor do we need one-hot encodings
    loss = torch.nn.CrossEntropyLoss()
    sgd = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=_LEARNING_RATE, momentum=0.9, weight_decay=1e-7)
    if _USE_SCHED:
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(sgd,'min',patience=2,verbose=True,threshold=0.0001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(sgd,milestones=[2,8,15])
    else:
        scheduler = None

    writer = SummaryWriter(_LOGDIR)

    try:
        run(model, train_loader, loss, sgd, writer, scheduler, test_loader=test_loader)
    except KeyboardInterrupt:
        answer = input("Do you want to save the model and the current running statistics? [y or n]\n")
        if answer == 'y':

            # find the accuracy before shutting
            acc = get_test_accuracy(model, test_loader)

            # model saving 
            if acc.data[0] > best_prec1:
                print("Saving this model as the best.")
                best_prec1 = acc.data[0]
                save_checkpoint(args, {'epoch': j + 1,'state_dict': model.state_dict(),'best_prec1': best_prec1}, True)

            # store the grads
            print("Storing the gradients for Tensorboard")
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # print("Histogram for[Name]: ",name)
                    writer.add_histogram(name, param.clone().cpu().data.numpy(),global_step+1)
                    writer.add_histogram(name + '/gradient', param.grad.clone().cpu().data.numpy(),global_step+1)

        else:
            print("Exiting without saving anything")

        # close the writer
        writer.close()
        print("Logged in: ",_LOGDIR)

        sys.exit()

    print("Logged in: ",_LOGDIR)


if __name__ == '__main__':
    main()
