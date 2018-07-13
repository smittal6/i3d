import torch.utils.data as data
import torch
import os
import math
import os.path
import numpy as np

from numpy.random import randint
from PIL import Image
from ReichardtDS8 import *
from retina_convert import retina
from opts import args
from utils import rescale

import warnings
warnings.simplefilter("error")

class VideoRecord(object):
    def __init__(self, row, modality):
        self._data = row

        for parent, dirnames, filenames in os.walk(self._data[0]):
            filenames = [f for f in filenames if '.avi' not in f]

        self.len = int(len(filenames)/3)
        # self.len = int(self.len/3) # 3 to account for img_, and 2 X flow_

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return self.len

    @property
    def label(self):
        return int(self._data[1])-1


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='rgb',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False, two_stream = False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.two_stream = two_stream
        self.rgb_format = 'img_{:05d}.jpg'

        # if self.modality in ['RGBDiff', 'EDR', 'GrayDiff', 'RGBEDR', 'MulEDR']:
            # self.new_length += 1# Diff needs one more image to calculate diff

        print("Called TSNDataset Init")

        self._parse_list()

        if self.modality == 'rgbdsc' or 'flowdsc':
            trans = []
            const = 2*math.pi/8.0
            for i in range(8):
                tup = [math.cos(i * const), math.sin(i * const)]
                trans.append(tup)

            # x component is the first
            self.basis = torch.from_numpy(np.asarray(trans)) # 8 rows and 2 columns

    def _load_image(self, directory, idx, override = False):
        '''
        Override represents the case when we have to load data for stream two, which will be rgb
        '''
        # print("Index: ",idx)
        if override or self.modality == 'rgb' or self.modality == 'RGBDiff' or self.modality == 'RGBEDR' or self.modality == 'MulEDR':
            # print("Entering override")
            # print(str(os.path.join(directory, self.rgb_format.format(idx))))
            ret = [Image.open(os.path.join(directory, self.rgb_format.format(idx))).convert('RGB')]
        elif self.modality == 'edr1' or self.modality == 'GrayDiff' or self.modality == 'rgbdsc' or self.modality == 'flyflow':
            # Grayscale
            ret = [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('L')]
        elif self.modality == 'flow' or self.modality == 'flowdsc':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')
            ret = [x_img, y_img]
        
        return ret

    def _parse_list(self):
        print("Parsing list now")
        self.video_list = [VideoRecord(x.strip().split(' '), self.modality) for x in open(self.list_file)]
        print("Found %d videos "%(len(self.video_list)))

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) / self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
            # print("Offsets: ",offsets)
        elif record.num_frames > self.num_segments:
            # offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            offsets = np.zeros((self.num_segments,))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # print("Obtained the record")

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        # print("Getting the items finally")

        try:
            # print("Trying to obtain sample")
            sample = self.get(record, segment_indices)
        except Exception as e:
            print("Exception encountered in __getitem__: \n",e)
            sample = None
        return sample


    def get(self, record, indices):

        # Stream 1 processing
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        # print("size after processing: ",process_data.size())

        # now depending on the modalilty of input, apply either Reichardt, or simple transform
        if self.modality == 'rgbdsc':

            out = process_data.numpy()
            # print("out's shape : ",out.shape)
            # This is a tuple right now
            vp1, vm1, vp2, vm2, vp3, vm3, vp4, vm4 = Reichardt8(out.transpose(1,2,3,0))
            # print("vp1's shape : ",vp1.shape)
            out = np.concatenate((vp1, vm1, vp2, vm2, vp3, vm3, vp4, vm4), axis=-1)

            # Convert the given matrix to have values in a certain range
            out = rescale(out, scale_min=-1,scale_max=1)
            process_data = torch.from_numpy(out).permute(3,0,1,2)
            # print("process_data: ",process_data.size())

        elif self.modality == 'flyflow':
            out = process_data.numpy()
            out_list = Reichardt8(out.transpose(1,2,3,0),args.rdirs)
            out = np.concatenate(out_list, axis=-1)
            out = rescale(out, scale_min=-1,scale_max=1)
            process_data = torch.from_numpy(out).permute(3,0,1,2)
            # print("process_data: ",process_data.size())
        
        elif self.modality == 'flowdsc':
            process_data = torch.matmul(self.basis, process_data.double().permute(1,2,0,3))
            process_data = process_data.permute(2,0,1,3).float()
            # print("Mean: %0.3f, Deviation: %0.3f"%(process_data.mean(),process_data.std()))

        elif self.modality == 'edr1':
            out = process_data.numpy()
            out = retina(out.transpose(1,2,3,0), alpha=0.5, mu_on=0.05,mu_off=-0.10)
            try:
                out = rescale(out, scale_min=-1,scale_max=1)
            except RuntimeWarning:
                print("Path of the video with zero error: ",record.path)
                exit()
            process_data = torch.from_numpy(out).permute(3,0,1,2)
            # print("process_data: ",process_data.size()) # Shape ~ [2,Timesteps,H,W]

        # print("processed data size: ",process_data.size())

        # Stream 2 processing
        # sec_images represents the images for second stream
        if self.two_stream is True:
            # print("Obtaining second stream data")
            sec_images = list()
            for seg_ind in indices:
                p = int(seg_ind)
                for i in range(self.new_length):
                    seg_imgs = self._load_image(record.path, p, override = True)
                    sec_images.extend(seg_imgs)
                    if p < record.num_frames:
                        p += 1
            dat = self.transform(sec_images)

            return process_data, dat, record.label

        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
