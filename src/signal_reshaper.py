# import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()


def reshape_signal(signal, signal_shape='naive-RGB'):
    shape, param = signal_shape.split('-')[0], signal_shape.split('-')[1]
    if shape == 'naive':
        s = np.array([[float(i)] for i in signal])
        s = scaler.fit_transform(s)
        s = np.array([[int(i * 255)] * 3 for i in s])
        s = s.reshape(36, 36, 3)

    return s


def reshape_signal2(signal, signal_shape='naive-RGB'):
    shape, param = signal_shape.split('-')[0], signal_shape.split('-')[1]
    if shape == 'naive':
        s = np.array([[float(i)] for i in signal])
        s = scaler.fit_transform(s)
        s = np.array([[int(i * 255)] * 3 for i in s])
        s = s.reshape(36, 36, 3)
    return s





