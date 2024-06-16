# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import scipy.signal as signal
from scipy import interpolate

class aceEmdDealt:
    def __init__(self, source_data: np.array):
        aceEmd = []
        for group_data in source_data:
            # print(group_data.shape)
            aceTmp = []
            for source_data in group_data.T:
                aceCurve = self.__empirical_decomposition__(source_data)
                aceTmp.append(aceCurve)
            aceEmd.append(np.array(aceTmp).T)
        self.aceEmd = np.array(aceEmd)

    def getAceResult(self):
        return self.aceEmd

    def __max_min_peaks__(self, data):
        point_num = np.size(data)
        peaks_max = signal.argrelextrema(data, np.greater)[0]
        peaks_min = signal.argrelextrema(data, np.less)[0]

        peaks_max = np.concatenate(([0], peaks_max, [point_num-1]))
        peaks_min = np.concatenate(([0], peaks_min, [point_num-1]))
        
        ext_peaks_max = np.delete(peaks_max, np.where(peaks_max[1:] == peaks_max[:-1]))
        ext_peaks_min = np.delete(peaks_min, np.where(peaks_min[1:] == peaks_min[:-1]))

        _tmp = np.sort(np.concatenate(([0], peaks_max, peaks_min, [point_num-1])))
        _tmp = np.delete(_tmp, np.where(_tmp[1:] == _tmp[:-1]))
        mid_point = []
        for i in range(_tmp.shape[0] - 1):
            mid_point.append(int((_tmp[i]+_tmp[i+1]) / 2))

        peaks_max = np.sort(np.concatenate((peaks_max, mid_point)))
        peaks_min = np.sort(np.concatenate((peaks_min, mid_point)))

        mid_peaks_max = np.delete(peaks_max, np.where(peaks_max[1:] == peaks_max[:-1]))
        mid_peaks_min = np.delete(peaks_min, np.where(peaks_min[1:] == peaks_min[:-1]))

        return ext_peaks_max, ext_peaks_min, mid_peaks_max, mid_peaks_min
    
    def __cubic_spline_3pts__(self, x, y, T):
        """
        Apparently scipy.interpolate.interp1d does not support
        cubic spline for less than 4 points.
        """
        x0, x1, x2 = x
        y0, y1, y2 = y

        x1x0, x2x1 = x1 - x0, x2 - x1
        y1y0, y2y1 = y1 - y0, y2 - y1
        _x1x0, _x2x1 = 1.0 / x1x0, 1.0 / x2x1

        m11, m12, m13 = 2 * _x1x0, _x1x0, 0
        m21, m22, m23 = _x1x0, 2.0 * (_x1x0 + _x2x1), _x2x1
        m31, m32, m33 = 0, _x2x1, 2.0 * _x2x1

        v1 = 3 * y1y0 * _x1x0 * _x1x0
        v3 = 3 * y2y1 * _x2x1 * _x2x1
        v2 = v1 + v3

        M = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
        v = np.array([v1, v2, v3]).T
        k = np.array(np.linalg.inv(M).dot(v))

        a1 = k[0] * x1x0 - y1y0
        b1 = -k[1] * x1x0 + y1y0
        a2 = k[1] * x2x1 - y2y1
        b2 = -k[2] * x2x1 + y2y1

        t = T
        t1 = (T[np.r_[T < x1]] - x0) / x1x0
        t2 = (T[np.r_[T >= x1]] - x1) / x2x1
        t11, t22 = 1.0 - t1, 1.0 - t2

        q1 = t11 * y0 + t1 * y1 + t1 * t11 * (a1 * t11 + b1 * t1)
        q2 = t22 * y1 + t2 * y2 + t2 * t22 * (a2 * t22 + b2 * t2)
        q = np.append(q1, q2)

        return t, q
    
    # envelopes
    def __envelopes__(self, data, peaks_max, peaks_min):
        point_num = len(data)

        if len(peaks_max) > 3:
            inp_max = interpolate.splrep(peaks_max, data[peaks_max], k=3)
            fit_max = interpolate.splev(np.arange(point_num), inp_max)
        else:
            _, fit_max = self.__cubic_spline_3pts__(peaks_max, data[peaks_max], np.arange(len(data)))

        if len(peaks_min) > 3:
            inp_min = interpolate.splrep(peaks_min, data[peaks_min], k=3)
            fit_min = interpolate.splev(np.arange(point_num), inp_min)
        else:
            _, fit_min = self.__cubic_spline_3pts__(peaks_min, data[peaks_min], np.arange(len(data)))

        return fit_max, fit_min
    
    def __imf_judge__(self, x: np.array, y: np.array):
        if (y.max() - y.min()) is not 0 and ((x - y)**2).sum() / (y.max() - y.min()) < 0.001:
            return True

        if not np.any(x == 0) and (((x - y) / x)**2).sum() < 0.2:
            return True

        if (y**2).sum() is not 0 and ((x - y)**2).sum() / (y**2).sum() < 0.2:
            return True

        return False

    def __empirical_decomposition__(self, source_data: np.array):
        # data
        data_ext = source_data.copy()
        data_mid = source_data.copy()
        # envelope line
        ext_up_envelopes, ext_down_envelopes = 0, 0
        mid_up_envelopes, mid_down_envelopes = 0, 0
        # extrema point
        ext_peaks_max, ext_peaks_min, mid_peaks_max, mid_peaks_min = self.__max_min_peaks__(data_mid)

        continue_time = 511
        # extrema emd & middle emd
        std_continue, old_std = 0, 0.0
        while True:
            if len(ext_peaks_max) < 3 or len(ext_peaks_min) < 3:
                break

            fit_max, fit_min = self.__envelopes__(data_mid, mid_peaks_max, mid_peaks_min)
            mid_up_envelopes, mid_down_envelopes = mid_up_envelopes + fit_max, mid_down_envelopes + fit_min
            data_mid = data_mid - (fit_max + fit_min) / 2

            _, _, mid_peaks_max, mid_peaks_min = self.__max_min_peaks__(data_mid)

            fit_max, fit_min = self.__envelopes__(data_ext, ext_peaks_max, ext_peaks_min)
            ext_up_envelopes, ext_down_envelopes = ext_up_envelopes + fit_max, ext_down_envelopes + fit_min
            data_ext_old = data_ext.copy()
            data_ext = data_ext - (fit_max + fit_min) / 2

            ext_peaks_max, ext_peaks_min, _, _ = self.__max_min_peaks__(data_ext)
            pass_zero = np.sum(data_ext[:-1] * data_ext[1:] < 0)

            std = abs((fit_max + fit_min) / 2 / source_data).mean()
            std_continue = (std_continue << 1) & continue_time
            std_continue += 1 if abs(old_std - std) < 1e-6 else 0
            old_std = std

            if (abs(pass_zero - len(ext_peaks_max) - len(ext_peaks_min)) < 2) or self.__imf_judge__(data_ext, data_ext_old) or std_continue == continue_time:
                break

        mid_curve = (ext_up_envelopes + ext_down_envelopes) / 4 + (mid_up_envelopes + mid_down_envelopes) / 4
        return mid_curve


class AceUnitData(Dataset):
    def __init__(self, data_set: pd.DataFrame, unit_size: int, predict_size: int, emd_col: list, result_col: list):
        """
        :param data_set: stock data set
        :param unit_size: the number of days in each unit
        :param predict_size: the number of predict days in each unit
        :param emd_col: column name of input data
        :param result_col: column name of result data
        :param emd_type: the emd type, 1->classic emd, 2->extreme emd, 3->middle emd (default)
        """
        super(AceUnitData, self).__init__()
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

        # normalized input data set
        # dealt by emd
        input_emd = aceEmdDealt(input_data)
        input_data, _, _ = max_min_normalised(input_emd.getAceResult()[:, :-predict_size])

        zero_shape = list(input_data.shape)
        zero_shape[1] = predict_size
        self.input_data = np.concatenate((input_data, np.zeros(zero_shape)), axis=1)

        # normalized result data set
        result_emd = aceEmdDealt(result_data)
        _, self.max_set, self.min_set = max_min_normalised(result_emd.getAceResult()[:, :-predict_size])
        result_data = (result_emd.getAceResult() - np.expand_dims(self.min_set, -1)) / np.expand_dims(self.max_set, -1)

        self.result_data = result_data
        
    def __len__(self):
        return self.unit_number

    def __getitem__(self, item: int):
        return self.input_data[item], self.stamp[item], self.result_data[item]

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