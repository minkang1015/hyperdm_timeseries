import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from data.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_SNP(Dataset):
    def __init__(self, root_path='./dataset',
                 flag='train', size=None, data_path='snp500.csv',
                 scale=True, timeenc=1, freq='d', 
                 train_end='2013-12-31', val_end='2015-12-31'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 5 * 4 * 12 
            self.label_len = 5 * 4
            self.pred_len = 5 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.train_end = train_end
        self.val_end = val_end

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_raw.rename(columns={'Date':'date'}, inplace=True)

        # if self.train_start is not None:
        #     df_raw = df_raw[df_raw['date'] >= self.train_start]

        # resample with frequency
        if self.freq == 'd':
            pass
        elif self.freq == 'w':
            # Convert date column to datetime format
            df_raw['date'] = pd.to_datetime(df_raw['date'])
            # Set date as index
            df_raw = df_raw.set_index('date')
            # Resample to weekly frequency and take the last day of each week
            df_raw = df_raw.resample('W').last()
            # Reset index to make date a column again
            df_raw = df_raw.reset_index()
        elif self.freq == 'm':
            # Convert date column to datetime format
            df_raw['date'] = pd.to_datetime(df_raw['date'])
            # Set date as index
            df_raw = df_raw.set_index('date')
            # Resample to monthly frequency and take the last day of each month
            df_raw = df_raw.resample('M').last()
            # Reset index to make date a column again
            df_raw = df_raw.reset_index()
            
        print(len(df_raw), 'raw data points')
        self.data_index = df_raw['date'].values
        
        num_train = df_raw[(df_raw['date'] >= '2005-01-01') & (df_raw['date'] <= self.train_end)].shape[0]
        num_vali = df_raw[(df_raw['date'] > self.train_end) & (df_raw['date'] <= self.val_end)].shape[0]
        num_test = df_raw[(df_raw['date'] > self.val_end)].shape[0]

        print(f"num_train: {num_train}, num_vali: {num_vali}, num_test: {num_test}")

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        self.train_index = df_raw['date'].values[border1s[0]:border2s[0]]
        self.valid_index = df_raw['date'].values[border1s[1]:border2s[1]]
        self.test_index  = df_raw['date'].values[border1s[2]:border2s[2]]

        self.train_index = self.train_index[self.seq_len + self.pred_len - 1 :]
        self.valid_index = self.valid_index[self.seq_len + self.pred_len - 1 :]
        self.test_index  = self.test_index[self.seq_len + self.pred_len - 1 :]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index 
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_DOW(Dataset):
    def __init__(self, root_path='./dataset',
                 flag='train', size=None, data_path='dow30.csv',
                 scale=True, timeenc=1, freq='d', 
                 train_end='2017-12-31', val_end='2018-12-31'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 5 * 4 * 12 
            self.label_len = 5 * 4
            self.pred_len = 5 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.train_end = train_end
        self.val_end = val_end

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_raw.rename(columns={'Date':'date'}, inplace=True)

        # if self.train_start is not None:
        #     df_raw = df_raw[df_raw['date'] >= self.train_start]

        self.data_index = df_raw['date'].values

        num_train = df_raw[(df_raw['date'] >= '2005-01-01') & (df_raw['date'] <= self.train_end)].shape[0]
        num_vali = df_raw[(df_raw['date'] > self.train_end) & (df_raw['date'] <= self.val_end)].shape[0]
        num_test = df_raw[(df_raw['date'] > self.val_end)].shape[0]

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        self.train_index = df_raw['date'].values[border1s[0]:border2s[0]]
        self.valid_index = df_raw['date'].values[border1s[1]:border2s[1]]
        self.test_index  = df_raw['date'].values[border1s[2]:border2s[2]]

        self.train_index = self.train_index[self.seq_len + self.pred_len - 1 :]
        self.valid_index = self.valid_index[self.seq_len + self.pred_len - 1 :]
        self.test_index  = self.test_index[self.seq_len + self.pred_len - 1 :]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        last_date = self.data_index[r_end - 1]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
