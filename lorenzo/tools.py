from river.drift import ADWIN, KSWIN
from compute_psi import calculate_psi
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def adwin_detection(data_stream):
    adwin = ADWIN()
    indexes = []
    for i, val in enumerate(data_stream):
        in_drift_adwin, _ = adwin.update(val)
        if in_drift_adwin:
          indexes.append(i)
    adwin.reset()
    return indexes

def kswin_detection(data_stream):
    kswin = KSWIN()
    indexes = []
    for i, val in enumerate(data_stream):
        in_drift_kswin, _ = kswin.update(val)
        if in_drift_kswin:
          indexes.append(i)
    kswin.reset()
    return indexes

def psi_detection(data_stream, w0, w1, slidind_window = True):
    indexes_high = []
    indexes_mid = []

    if slidind_window:
        for i in range(w0 + w1, len(data_stream)):
            val = calculate_psi(data_stream[i - w1 - w0: i - w0], data_stream[i - w0:i])
            if val >= 0.3:
                indexes_high.append(i)
            else:
                if val >= 0.2:
                    indexes_mid.append(i)
    else:
        for i in range(w0, len(data_stream)):
            val = calculate_psi(data_stream[: w0], data_stream[i - w1:i])
            if val >= 30:
                indexes_high.append(i)
            else:
                if val >= 20:
                    indexes_mid.append(i)

    return indexes_high, indexes_mid

def outlier_test(date, data, q_1=0.25, q_2=0.75):
    q1 = np.quantile(data,q_1)
    q2 = np.quantile(data, q_2)
    iqr = q2 - q1
    outlier_bool = np.logical_or(date < (q1 - 1.5 * iqr), date > (q2 + 1.5 * iqr))
    return outlier_bool

def outliers_detection(data_stream, w0):
    index = []
    for el in range(w0,len(data_stream)):
      is_outlier = outlier_test(data_stream[el], data_stream[el - w0:el], q_1 = 0.25, q_2 = 0.75)
      if is_outlier:
        index.append(el)
    return index

def show_plot(data_stream, detected_indexes, real_indexes = None, syntetic_dataset = True ):

    if syntetic_dataset:
        plt.figure(figsize=(15, 5))
        plt.plot(np.arange(len(data_stream)), data_stream)
        for el in detected_indexes:
            plt.axvline(el, color='red', linestyle='--')
        if real_indexes:
            for el in real_indexes:
                plt.axvline(el, color='green', linestyle='--')
    else:
        temp = data_stream.iloc[detected_indexes].index
        plt.figure(figsize=(15, 5))
        plt.plot(data_stream.index, data_stream)
        for date in temp:
            plt.axvline(datetime(date.year, date.month, date.day), color='red', linestyle='--')

