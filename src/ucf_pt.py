# __author__ Siddharth Mittal
import torch
import time
import sys
import torchvision

from tensorboardX import SummaryWriter
from i3dmod import modI3D   # i3d model
from utils import *
from transforms import *    # dataset transforms
from dataset import TSNDataSet
from opts import args    # options(args)

# ../model/model_rgb.pth
# args = parser.parse_args()
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


print("Finetune: ",str(_FT))
_LOGDIR = '../ftlogs/' + _MODALITY + '/' + _WTS + '_' + str(_LEARNING_RATE) + '_' + str(_EPOCHS)

if args.nstr is not None:
    _LOGDIR = _LOGDIR + "_" + args.nstr


def get_set_loader():

    # i_h = 240
    # i_w = 320

    # Take the min value, and use it for the ratio
    # min_ = min(i_w, i_h)
    # ratio = float(min_) / 256

    # new_width = int(float(i_w) / ratio)
    # new_height = int(float(i_h) / ratio)
    # print("W: %d, H: %d"%(new_width, new_height))

    pure = True if _MODALITY == 'rgb' or _MODALITY == 'flow' else False
    modlist = ["rgb", "rgbdsc", "flyflow", "edr1"]

    train_transform = torchvision.transforms.Compose(
        [GroupScale(size=256),GroupRandomCrop(size=_IMAGE_SIZE), Stack(modality=_MODALITY), ToTorchFormatTensor(pure=pure)])

    test_transform = torchvision.transforms.Compose(
        [GroupScale(size=256), GroupCenterCrop(size=_IMAGE_SIZE), Stack(modality=_MODALITY), ToTorchFormatTensor(pure=pure)])

    train_dataset = TSNDataSet("", _TRAIN_LIST, num_segments=1, new_length=64, modality=_MODALITY, 
                               image_tmpl="img_{:05d}.jpg" if _MODALITY in modlist else "flow_"+"{}_{:05d}.jpg",
                               transform=train_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_BATCH_SIZE, shuffle=True, num_workers=_NUM_W, collate_fn=my_collate)

    test_dataset = TSNDataSet("", _TEST_LIST, num_segments=1, new_length=64, modality=_MODALITY, 
                              image_tmpl="img_{:05d}.jpg" if _MODALITY in modlist else "flow_"+"{}_{:05d}.jpg",
                              transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_TEST_BATCH_SIZE, shuffle=True, num_workers=_NUM_W, collate_fn=my_collate)

    return train_loader, test_loader

def run(model, train_loader, criterion, optimizer, train_writer, scheduler, test_loader=None):

    avg_loss = AverageMeter()
    train_acc = AverageMeter()

    global j
    global global_step
    global best_prec1

    train_points = len(train_loader)
    global_step = 1
    best_prec1 = 0.0

    for j in range(_EPOCHS):

        print("Epoch Number: %d" % (j + 1))
        if scheduler is not None:
            scheduler.step()
            for param_group in optimizer.param_groups:
                print("Learning Rate: ",param_group['lr'])

        avg_loss.reset()
        train_acc.reset()

        for i, (input_3d, target) in enumerate(train_loader):

            # print("Target shape: ",target.size())

            # Prepare data for pytorch forward pass
            start = time.time()

            input_3d_var = torch.autograd.Variable(input_3d.cuda())
            if args.thres is not None:
                input_3d_var = torch.nn.functional.threshold(input_3d_var,threshold=args.thres,value=0.0)
            target = torch.autograd.Variable(target.cuda())

            # Pytorch forward pass
            # print("Input shape: ",input_3d_var.size())
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

            
            optimizer.zero_grad()
            global_step += 1

        # scheduler.step(avg_loss.avg)
        if test_loader is not None:
            acc = get_test_accuracy(model, test_loader)
            train_writer.add_scalar('Test Acc', acc.data[0], global_step)
            print("Best Accuracy till now: %0.5f " % best_prec1)
            if acc.data[0] > best_prec1:
                print("Saving this model as the best.")
                best_prec1 = acc.data[0]
                print("Best Accuracy till now: %0.5f " % best_prec1)
                save_checkpoint(args, {'epoch': j + 1,'state_dict': model.state_dict(),'best_prec1': best_prec1}, True)

    get_test_accuracy(model, test_loader)
    train_writer.close()


def main():
    global j
    global global_step
    global best_prec1

    train_loader, test_loader = get_set_loader()

    # args order: modality, num_c, finetune, dropout
    model = modI3D(modality=_MODALITY, wts=_WTS, dog=args.dog, mean=args.mean, random=args.random)

    # if args.two_stream is not False:
        # model_stream2 = modI3D(modality='rgb', wts='rgb', dog=args.dog2)
    
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
        scheduler = torch.optim.lr_scheduler.MultiStepLR(sgd,milestones=args.lr_steps)
    else:
        scheduler = None

    writer = SummaryWriter(_LOGDIR)

    try:
        run(model, train_loader, loss, sgd, writer, scheduler, test_loader=test_loader)
    except KeyboardInterrupt:
        answer = input("Do you want to save the model and the current running statistics? [y or n]\n")
        if answer == 'y':
            interruptHandler(args, model, writer, test_loader, best_prec1, global_step, j)
        else:
            print("Exiting without saving anything")

        # close the writer
        writer.close()
        print("Logged in: ",_LOGDIR)
        sys.exit()

    print("Logged in: ",_LOGDIR)

if __name__ == '__main__':
    main()
