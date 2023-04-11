# -*- coding: utf-8 -*-
import myemd
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import scipy.signal as signal
from scipy import interpolate
from scipy.signal import argrelextrema


class EMD_dealt:
    def __init__(self, source_data: np.array, emd_type: int=3, imf_times: int=10):
        '''
        :param source_data: three-dimensional data, respectively, batch size, number of days, data per stock
        :param emd_type: the emd type, 1->iceemd, 2->eceemd, 3->aceemd (default)
        '''
        #  Gaussian noise
        noise_list = []
        win_len = source_data.shape[-2]
        for _ in range(imf_times // 2):
            noise = np.random.randn(win_len)
            n_up_envelopes, n_down_envelopes = myemd.emd(noise)
            noise_list.append((n_up_envelopes + n_down_envelopes) / 2 / np.std(noise))

            n_up_envelopes, n_down_envelopes = myemd.emd(-noise)
            noise_list.append((n_up_envelopes + n_down_envelopes) / 2 / np.std(-noise))

        # emd process
        emd_result = []
        for s in range(len(source_data)):
            emd_tmp = []
            for d in range(len(source_data[s].T)):
                _data = (source_data[s].T)[d]
                # envelope line
                up_list, down_list = [], []
                # iceemd & eceemd
                if emd_type == 1 or emd_type == 2:
                    for noise in noise_list:
                        _emd_data = _data.copy() + noise * myemd.snr(_data, noise)
                        up, down = myemd.emd(_emd_data) if emd_type == 1 else myemd.extemd(_emd_data)

                        up_list.append(up)
                        down_list.append(down)
                # aceemd
                else:
                    for i in range(imf_times // 2):
                        _exemd_data = _data.copy() + noise_list[2*i] * myemd.snr(_data, noise_list[2*i])
                        _acemd_data = _data.copy() + noise_list[2*i+1] * myemd.snr(_data, noise_list[2*i+1])
                        up, down = myemd.aceemd(_exemd_data, _acemd_data, 0.3)

                        up_list.append(up)
                        down_list.append(down)
                # denoise
                emd_tmp.append((np.array(up_list).mean(axis=0) + np.array(down_list).mean(axis=0)) / 2)
            emd_result.append(np.array(emd_tmp).T)
        self.emd_result = np.array(emd_result)

    def getEmdResult(self):
        return self.emd_result


class EmdData(Dataset):
    def __init__(self, data_set: pd.DataFrame, unit_size: int, predict_size: int, emd_col: list, result_col: list, emd_type: int = 3):
        """
        :param data_set: stock data set
        :param unit_size: the number of days in each unit
        :param predict_size: the number of predict days in each unit
        :param emd_col: column name of input data
        :param result_col: column name of result data
        :param emd_type: the emd type, 1->iceemd, 2->eceemd, 3->aceemd (default)
        """
        super(EmdData, self).__init__()
        self.unit_number = int(data_set.shape[0] - unit_size + 1)

        # global position
        time_stamp = data_set[['trade_date']].copy().reset_index(drop=True)
        # calculate the number of month, weekday and date
        time_stamp['trade_date'] = pd.to_datetime(time_stamp.trade_date, format='%Y%m%d')
        m, w, d = [], [], []
        for df_i in range(len(time_stamp)):
            tmp = time_stamp.trade_date[df_i]
            m.append(tmp.month)
            w.append(tmp.weekday())
            d.append(tmp.day)
        time_stamp['month'], time_stamp['weekday'], time_stamp['day'] = m, w, d
        time_stamp = time_stamp.drop(['trade_date'], axis=1).values

        # create unit time
        source_time = []
        for unit_i in range(self.unit_number):
            source_time.append(time_stamp[unit_i: unit_i + unit_size])
        self.stamp = np.array(source_time).astype(int)

        # stock data
        emd_set = data_set.get(emd_col).values
        result_set = data_set.get(result_col).values
        # create unit data
        input_data, result_data = [], []
        for unit_i in range(self.unit_number):
            input_data.append(emd_set[unit_i: unit_i + unit_size])
            result_data.append(result_set[unit_i: unit_i + unit_size])
        input_data = np.array(input_data).astype(float)
        result_data = np.array(result_data).astype(float)

        # normalized emd input data
        input_emd = EMD_dealt(input_data[:, :-predict_size], emd_type)
        input_data, _, _ = max_min_normalised(input_emd.getEmdResult())

        # normalized emd result data
        result_emd = EMD_dealt(result_data, emd_type)
        result_data, self.max_set, self.min_set = max_min_normalised(result_emd.getEmdResult())

        # set input and result data
        zero_shape = list(input_data.shape)
        zero_shape[1] = predict_size
        self.input_data = np.concatenate((input_data, np.zeros(zero_shape)), axis=1)
        self.result_data = result_data

    def __len__(self):
        return self.unit_number

    def __getitem__(self, item: int):
        return self.input_data[item], self.stamp[item], self.result_data[item]

    def anti_normalize_data(self, true: np.array, predict: np.array):
        anti_true = true * self.max_set + self.min_set
        anti_predict = predict * self.max_set.reshape(-1,1) + self.min_set.reshape(-1,1)
        return anti_true, anti_predict


class EndToEndData(Dataset):
    def __init__(self, data_set: pd.DataFrame, unit_size: int, predict_size: int, model_col: list, result_col: list):
        """
        :param data_set: data set
        :param unit_size: the number of days for a unit
        :param predict_size: the number of days for predict days in each unit
        :param data_col: transformer column
        """
        super(EndToEndData, self).__init__()
        self.unit_number = int(data_set.shape[0] - unit_size + 1)

        model_set = data_set.get(model_col).values
        time_stamp = data_set[['trade_date']].copy().reset_index(drop=True)
        result_set = data_set.get(result_col).values

        # calculate the number of month, weekday and date
        time_stamp['trade_date'] = pd.to_datetime(time_stamp.trade_date, format='%Y%m%d')
        m, w, d = [], [], []
        for df_i in range(len(time_stamp)):
            tmp = time_stamp.trade_date[df_i]
            m.append(tmp.month)
            w.append(tmp.weekday())
            d.append(tmp.day)
        time_stamp['month'], time_stamp['weekday'], time_stamp['day'] = m, w, d
        time_stamp = time_stamp.drop(['trade_date'], axis=1).values

        # create unit data and time
        source_time, source_data, result_data = [], [], []
        for unit_i in range(self.unit_number):
            source_time.append(time_stamp[unit_i: unit_i + unit_size])
            source_data.append(model_set[unit_i: unit_i + unit_size])
            result_data.append(result_set[unit_i: unit_i + unit_size])
        self.stamp = np.array(source_time).astype(int)
        source_data = np.array(source_data).astype(float)
        result_data = np.array(result_data).astype(float)

        # normalized data set
        source_data, _, _ = max_min_normalised(source_data[:, :-predict_size])
        # self.input = source_data[:, : -predict_size]
        zero_shape = list(source_data.shape)
        zero_shape[1] = predict_size
        self.input = np.concatenate((source_data, np.zeros(zero_shape)), axis=1)

        # normalized result data set
        self.label, self.max_set, self.min_set = max_min_normalised(result_data)

    def anti_normalize_data(self, true: np.array, predict: np.array):
        # anti normalize true data
        anti_true = true * self.max_set + self.min_set
        # anti normalize predict data
        anti_predict = predict * self.max_set.reshape(-1,1) + self.min_set.reshape(-1,1)
        
        return anti_true, anti_predict

    def __len__(self):
        return self.unit_number

    def __getitem__(self, item: int):
        return self.input[item], self.stamp[item], self.label[item]


class FormerData(Dataset):
    def __init__(self, data_set: pd.DataFrame, unit_size: int, predict_size: int, model_col: list, result_col: list):
        """
        :param data_set: stock data set
        :param unit_size: the number of days in each unit
        :param label_size: the number of days for decoder input feature
        :param predict_size: the number of predict days in each unit
        :param model_col: column name of input data
        :param result_col: column name of result data
        """
        super(FormerData, self).__init__()
        self.unit_number = int(data_set.shape[0] - unit_size + 1)

        # global position
        time_stamp = data_set[['trade_date']].copy().reset_index(drop=True)
        # calculate the number of month, weekday and date
        time_stamp['trade_date'] = pd.to_datetime(time_stamp.trade_date, format='%Y%m%d')
        m, w, d = [], [], []
        for df_i in range(len(time_stamp)):
            tmp = time_stamp.trade_date[df_i]
            m.append(tmp.month)
            w.append(tmp.weekday())
            d.append(tmp.day)
        time_stamp['month'], time_stamp['weekday'], time_stamp['day'] = m, w, d
        time_stamp = time_stamp.drop(['trade_date'], axis=1).values

        # create unit time
        source_time = []
        for unit_i in range(self.unit_number):
            source_time.append(time_stamp[unit_i: unit_i + unit_size])
        stamp = np.array(source_time).astype(int)
        # time array
        source_time = np.array(source_time).astype(int)
        self.en_stamp = source_time[:, :-predict_size]
        self.de_stamp = source_time[:, predict_size:]


        # stock data
        model_set = data_set.get(model_col).values
        result_set = data_set.get(result_col).values
        # create unit data and time
        model_data, result_data = [], []
        for unit_i in range(self.unit_number):
            model_data.append(model_set[unit_i: unit_i + unit_size])
            result_data.append(result_set[unit_i: unit_i + unit_size])
        model_data = np.array(model_data).astype(float)
        result_data = np.array(result_data).astype(float)

        # normalized encode data set
        self.en_input, _, _ = max_min_normalised(model_data[:, :-predict_size])
        
        # create encoder and decoder input data set
        de_input, self.max_set, self.min_set = max_min_normalised(result_data[:, predict_size: -predict_size])

        zero_shape = list(result_data.shape)
        zero_shape[1] = predict_size
        self.de_input = np.concatenate((de_input, np.zeros(zero_shape)), axis=1)
        # normalized decode data set
        self.de_output = (result_data[:, predict_size:] - np.expand_dims(self.min_set, -1)) / np.expand_dims(self.max_set, -1)

    def __len__(self):
        return self.unit_number

    def __getitem__(self, item: int):
        return self.en_input[item], self.de_input[item], self.en_stamp[item], self.de_stamp[item], self.de_output[item]

    def anti_normalize_data(self, true: np.array, predict: np.array):
        # anti normalize true data
        anti_true = true * self.max_set + self.min_set
        # anti normalize predict data
        anti_predict = predict * self.max_set.reshape(-1,1) + self.min_set.reshape(-1,1)
        
        return anti_true, anti_predict


def max_min_normalised(data: np.array):
    """
    Maximum and minimum normalization
    :param data: data that needs to be normalized
    :return:
        max_set: maximum data in each unit
        min_set: maximum data in each unit
        normal_data: normalized data
    """
    unit_num = data.shape[0]

    max_set = []
    min_set = []
    normal_data = []

    for col_i in range(unit_num):
        data_i = data[col_i]

        min_set.append(np.min(data_i, axis=0))
        data_i = data_i - np.min(data_i, axis=0)
        max_set.append(np.max(data_i, axis=0))
        data_i = data_i / np.max(data_i, axis=0)

        normal_data.append(data_i)

    return np.array(normal_data), np.array(max_set), np.array(min_set)
