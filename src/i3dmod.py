import torch
import math
import numpy as np

from i3dpt import I3D, Unit3Dpy
from utils import kernels
from opts import args

class modI3D(torch.nn.Module):

    def __init__(self, modality, wts, dog, mean=False, random=False):

        super(modI3D, self).__init__()

        # Have to be different
        self.modality = modality
        self.weights = wts
        self.dog = dog

        # May be common 
        self.mean = mean
        self.random = random

        # Have to be common
        self.ft = args.ft

        self.transform = False
        self.path = ''
        self.num_c = 101
        self.center_surround = None

        # Initialize pytorch I3D
        self.i3d = I3D(num_classes=400, dropout_prob=0.5, modality=self.weights)

        # Define the final layers
        self.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))
        self.dropout = torch.nn.Dropout(0.5)
        self.conv3d_0c_1x1 = Unit3Dpy(in_channels=1024, out_channels=self.num_c, kernel_size=(1, 1, 1), activation=None, use_bias=True, use_bn=False)
        self.softmax = torch.nn.Softmax(1)
        self.configure()

    def configure(self):

        if self.weights == 'rgb':
            self.path = '../model/model_rgb.pth'
        else:
            self.path = '../model/model_flow.pth'

        if self.modality == 'rgb':
            self.in_channels = 3

        elif self.modality == 'flow':
            self.in_channels = 2

        elif self.modality == 'edr1':
            self.in_channels = 2
            self.transform = True

        elif self.modality == 'flyflow':
            # pick only these directions [Note for all 8 directions currently use flowdsc or rgbdsc]
            self.in_channels = len(args.rdirs)

            # We have to do this because we cannot transform the weight directly using the matrix
            # Also, currently no plan to use rgb weights with this, else we can do without transforming with 3 channels
            if len(args.rdirs) > 2:
                self.transform = True
                self.mean = True

        else:
            self.in_channels = 8
            self.transform = True

        if args.load():
            self.load_weights()

        if self.weights == 'rgb':
            print("Overriding the provided option for mean as using rgb weights [Can't transform weights from 3 dim space]")
            self.mean = True

        if self.transform:
            # without calling this the weights won't adapt
            self.adapt()

        if self.dog:
            self.assign_dog(_size=5)

        if self.ft is False:
            print("Setting grads as false")
            self.set_grads_false()


        # move the i3d model to cuda
        self.i3d.cuda()
        print("Pretrained i3D shifted to CUDA")

    def assign_dog(self, _size = 5):
        # 1 along time dimension to ensure this acts like a Conv2d

        # This padding is to ensure every pixel is centered at the filter window
        _pad=int((_size-1)/2)

        # in_channels, in_channels because the output will have the same shape, groups = in_channels because we want to apply the filter to each channel individually
        self.center_surround = torch.nn.Conv3d(self.in_channels, self.in_channels, kernel_size=(1,_size,_size), stride=1, padding=(0,_pad,_pad), bias=False, groups=self.in_channels)

        # repeat has 1 as second dimension as we require: (in_channels/groups), but groups = in_channel
        self.center_surround.weight = torch.nn.parameter.Parameter(torch.from_numpy(kernels(size=_size)).view(1,_size,_size).repeat(self.in_channels, 1, 1, 1, 1).float(),requires_grad=False)


    def adapt(self):
        '''
        Adapts the weights of the first convolutional layer in accordance with the number of channels in input
        :return:
        '''

        old_inchannels = self.i3d.conv3d_1a_7x7.conv3d.in_channels
        weight_3d = self.i3d.conv3d_1a_7x7.conv3d.weight.data
        # bias_3d = self.i3d.conv3d_1a_7x7.conv3d.bias

        new_weight_3d = weight_3d.mean(1)
        new_weight_3d = new_weight_3d.unsqueeze(1).repeat(1, self.in_channels, 1, 1, 1)

        if self.random:
            new_weight_3d = torch.randn(new_weight_3d.size())

        elif self.mean:
            # Modifier 1: Essentially this is the one without transform on the weights. Just the mean.
            new_weight_3d = new_weight_3d * old_inchannels / self.in_channels
        else:
            trans = []
            const = 2*math.pi/8.0
            for i in range(8):
                tup = [math.cos(i * const), math.sin(i * const)]
                trans.append(tup)

            # x component is the first, Shape of this is (8,2)
            trans = torch.from_numpy(np.asarray(trans))

            sizes = weight_3d.size()

            # TODO: make this efficient by using torch.matmul()
            for k_num in range(sizes[0]):
                for i in range(sizes[2]):
                    for j in range(sizes[3]):
                        for k in range(sizes[4]):
                            # : in new_weight_3d will represent the 8 channels
                            new_weight_3d[k_num,:,i,j,k] = torch.mm(trans,weight_3d[k_num,:,i,j,k].unsqueeze(1).double())

            print("New weight shape: ",new_weight_3d.size())

        if old_inchannels != self.in_channels:
            conv3d_1a_7x7 = Unit3Dpy(
                out_channels=64,
                in_channels=self.in_channels,
                kernel_size=(7, 7, 7),
                stride=(2, 2, 2),
                padding='SAME',
                use_bias=True)
            conv3d_1a_7x7.weight = torch.nn.parameter.Parameter(new_weight_3d)
            self.i3d.conv3d_1a_7x7 = conv3d_1a_7x7
        print("Weights Transformed")

    def load_weights(self):

        self.i3d.eval()
        self.i3d.load_state_dict(torch.load(self.path))
        print("Pretrained i3D weights restored")

        # print("Shape of weights: ",self.i3d.conv3d_1a_7x7.conv3d.weight.data.size())
        # i3nception_pt.train()

    def set_grads_false(self):
        for name, param in self.i3d.named_parameters():
            # print("Setting requires_grad as False for: ", name)
            param.requires_grad = False

    def forward(self, inp):
        # Loaded i3D section

        if self.dog:
            inp = self.center_surround(inp)
            # print("Post DoG shape: ",inp.size())

        # if self.noise is not None:
            # inp = inp + self.noise
        # if args.noise is not None:
            # noise = Variable(torch.randn(int(args.batch/args.gpus),self.in_channels,64,224,224).cuda().normal_(std=args.noise),requires_grad=False)
            # inp = inp + noise

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
