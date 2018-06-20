import torch
import math
import numpy as np

from i3dpt import I3D
from i3dpt import Unit3Dpy


class modI3D(torch.nn.Module):

    def __init__(self, modality='rgb', num_c=101, finetune=False, dropout_prob=0.5):

        super(modI3D, self).__init__()

        self.num_c = num_c

        # Initialize pytorch I3D
        if modality=='dsc':
            self.i3d = I3D(num_classes=400, dropout_prob=0.5, modality='flow')
        else:
            self.i3d = I3D(num_classes=400, dropout_prob=0.5, modality=modality)
        # Can't change the classes because weight loading will cause issues.

        # Define the final layers
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

        transform = False

        if modality == 'rgb':
            self.in_channels = 3
            self.path = '../model/model_rgb.pth'
        elif modality == 'flow':
            self.in_channels = 2
            self.path = '../model/model_flow.pth'
        elif modality == 'dsc':
            self.in_channels = 8
            self.path = '../model/model_flow.pth'
            transform = True

        self.load_weights(self.path)

        if transform:
            self.adapt(in_channels=self.in_channels)


        if finetune is False:
            print("Setting grads as false")
            self.set_grads_false()

        self.i3d.cuda()
        print("Pretrained i3D shifted to CUDA")

    def adapt(self, in_channels=3, mean=False):
        '''
        Adapts the weights of the first convolutional layer in accordance with the number of channels in input
        :return:
        '''

        old_inchannels = self.i3d.conv3d_1a_7x7.conv3d.in_channels
        weight_3d = self.i3d.conv3d_1a_7x7.conv3d.weight.data
        # bias_3d = self.i3d.conv3d_1a_7x7.conv3d.bias

        new_weight_3d = weight_3d.mean(1)
        new_weight_3d = new_weight_3d.unsqueeze(1).repeat(1, in_channels, 1, 1, 1)

        if mean:
            # Modifier 1: Essentially this is the one without transform on the weights. Just the mean.
            new_weight_3d = new_weight_3d * old_inchannels / in_channels
        else:
            trans = []
            const = 2*math.pi/8.0
            for i in range(8):
                tup = [math.cos(i * const), math.sin(i * const)]
                trans.append(tup)

            # x component is the first
            trans = torch.from_numpy(np.asarray(trans))
            # Shape of this is (8,2)

            sizes = weight_3d.size()
            for k_num in range(sizes[0]):
                for i in range(sizes[2]):
                    for j in range(sizes[3]):
                        for k in range(sizes[4]):
                            # : in new_weight_3d will represent the 8 channels
                            new_weight_3d[k_num,:,i,j,k] = torch.mm(trans,weight_3d[k_num,:,i,j,k].unsqueeze(1).double())

            print("New weight shape: ",new_weight_3d.size())

        if old_inchannels != in_channels:
            conv3d_1a_7x7 = Unit3Dpy(
                out_channels=64,
                in_channels=in_channels,
                kernel_size=(7, 7, 7),
                stride=(2, 2, 2),
                padding='SAME')
            conv3d_1a_7x7.weight = torch.nn.parameter.Parameter(new_weight_3d)
            self.i3d.conv3d_1a_7x7 = conv3d_1a_7x7

        print("Weights Transformed")

    def load_weights(self, weight_path):

        self.i3d.eval()
        self.i3d.load_state_dict(torch.load(weight_path))
        print("Pretrained i3D weights restored")

        # print("Shape of weights: ",self.i3d.conv3d_1a_7x7.conv3d.weight.data.size())
        # i3nception_pt.train()

    def set_grads_false(self):
        for name, param in self.i3d.named_parameters():
            # print("Setting requires_grad as False for: ", name)
            param.requires_grad = False

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
