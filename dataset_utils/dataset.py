import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from dataset_maker.eegLoader import EDFReader, MontageBuilder
from pathlib import Path
from dataset_maker.Config import *


class FileData(Dataset):

    def __init__(self, file_path, montageDict, window_size, overlap=0.0, l_freq=0.5, h_freq=80.0, rsfreq=125, clip=500, norm_method=None, task='multi_cls'):
        self.file_path = file_path
        self.montageDict = montageDict
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.rsfreq = rsfreq
        self.__clip = clip
        self.__norm_method = norm_method
        self.__window_size = window_size
        self.__stride_size = int(self.__window_size * (1 - overlap))
        self.task = task
        self.__read_data__()

    def __read_data__(self):

        montage_type = self.get_montage_type(self.file_path)

        reader = EDFReader(self.file_path, montage_type, self.montageDict, self.l_freq, self.h_freq, self.rsfreq)
        self.chOrder = list(reader.processed_data.columns)
        self.eegData = reader.processed_data.values.T
        self.label_table = self.parsing_label_from_file()

        data_len = self.eegData.shape[1]
        self.label = self.label_parsing()
        self.eegData = self.preprocess_data(self.eegData)

        # total number of samples
        self.total_sample_num = (data_len - self.__window_size) // self.__stride_size + 1

        return None

    def preprocess_data(self, subject):
        if self.__clip != np.inf:
            data = np.clip(self.eegData, -self.__clip, self.__clip)
            data = self.norm_data(data)
        return torch.from_numpy(data)

    def norm_data(self, data):
        if self.__norm_method in [None, '']:
            return data
        elif self.__norm_method == 'mm_scaling':
            return data / self.__clip
        elif self.__norm_method == 'z_score_scaling':
            return (data - np.mean(data, axis=-1, keepdims=True)) / np.std(data, axis=-1, keepdims=True)

    def label_parsing(self):

        c, length = self.eegData.shape
        label_table = self.label_table.values
        channel_index, label_index = label_table[:, 0], label_table[:, 3]
        start_index, stop_index = label_table[:, 1], label_table[:, 2]

        if self.task == 'binary_cls':

            # label : [N, T]
            label = np.zeros((c, length, 1))
            for idx in range(len(start_index)):
                channel_list = channel_index[idx]
                start = start_index[idx]
                stop = stop_index[idx]
                for channel_id in channel_list:
                    label[channel_id, start:stop] = 1

        if self.task == 'multi_cls':

            # label : [N, T, num_class]
            label = np.zeros((c, length, 6))
            for idx in range(len(start_index)):
                channel_list = channel_index[idx]
                l = label_index[idx]
                start = start_index[idx]
                stop = stop_index[idx]
                for channel_id in channel_list:
                    for ll in SINGLE_LABEL_TO_MULTI_LABEL[l]:
                        label[channel_id, start:stop, ll] = 1
        return torch.from_numpy(label)

    def get_montage_type(self, edfFile):
        return edfFile.parent.name[7:]

    @property
    def ch_names(self):
        return self.chOrder

    def __getitem__(self, idx):
        item_start_idx = idx * self.__stride_size
        data = self.eegData[:, item_start_idx:item_start_idx + self.__window_size]
        label = self.label[:, item_start_idx:item_start_idx + self.__window_size]
        return {'data': data, 'label': label}

    def __len__(self):
        return self.total_sample_num

    def parsing_label_from_file(self):

        labelFile = self.file_path.parent / self.file_path.name.replace('.edf', '.csv')
        label_table = pd.read_csv(labelFile, skiprows=6)

        label_table['label'] = label_table['label'].apply(lambda x: ARTIFACT_TO_ID[x])
        label_table[['start_time', 'stop_time']] = label_table[['start_time', 'stop_time']].apply(
            lambda x: np.int64(x * self.rsfreq))
        label_table.drop('confidence', axis=1, inplace=True)
        label_table.sort_values(['start_time', 'stop_time'], inplace=True)

        label_table['channel'] = label_table['channel'].apply(lambda x: CHANNEL_TO_ID[x])

        def multi_channel_to_int_coding(x):
            word = []
            for i in range(22):
                if i in x:
                    word.append('1')
                else:
                    word.append('0')
            return int(''.join(word), 2)

        label_table = label_table.groupby(['start_time', 'stop_time']).agg(
            {'channel': lambda x: list(x), 'label': lambda x: set(x).pop()})
        label_table.reset_index(inplace=True)
        label_table = label_table[['channel', 'start_time', 'stop_time', 'label']]
        label_table.columns = ['#Channel', 'start', 'end', 'label']

        return label_table

if __name__ == '__main__':

    montagePath = Path(MONTAGE_PATH)
    montageDict = MontageBuilder(path=montagePath).initial_montage()
    file_path = Path('/root/autodl-pub/YYF/WaveNet/data/01_tcp_ar/aaaaaguk_s002_t001.edf')
    file_data = FileData(file_path, montageDict, window_size=625, overlap=0.)

    pass