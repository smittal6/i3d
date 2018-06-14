import torch.utils.data as data
import torch

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

        for parent, dirnames, filenames in os.walk(self._data[0]):
            filenames = [f for f in filenames if '.avi' not in f]

        self.len = len(filenames)

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
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality in ['RGBDiff', 'EDR', 'GrayDiff', 'RGBEDR', 'MulEDR']:
            self.new_length += 1# Diff needs one more image to calculate diff

        print("Called TSN Init")

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff' or self.modality == 'RGBEDR' or self.modality == 'MulEDR':
            # print(os.path.join(directory, self.image_tmpl.format(idx)))
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'EDR' or self.modality == 'GrayDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('L')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        print("Parsing list now")
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

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

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = []
        for img in images:
            temp = self.transform(img) # len(images) = num_seg * data_length i.e. 7*6 =24
            process_data.append(temp)

        out = torch.stack(process_data,dim=0).transpose(0,1)
        # print("out shape: ",out.shape)
        return out, record.label

    def __len__(self):
        return len(self.video_list)
