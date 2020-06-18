# import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler




scaler = MinMaxScaler((0,255))


def reshape_signal(signal, signal_shape='naive-RGB'):
    shape, param = signal_shape.split('-')[0], signal_shape.split('-')[1]
    if shape == 'naive':
        s = np.array([[float(i)] for i in signal])
        s = scaler.fit_transform(s)
        s = s.astype(int)
        s = s.reshape(36, 36)
        s = [s, s, s]
        s = np.moveaxis(s, 0, 2)
    return s

def reshape_signal2(signal, signal_shape='naive-RGB'):
    shape, param = signal_shape.split('-')[0], signal_shape.split('-')[1]
    if shape == 'naive':
        s = np.array([[float(i)] for i in signal])
        s = scaler.fit_transform(s)
        s = s.astype(int)
        s = s.reshape(36, 36)
        s = [s, s, s]
        s = np.moveaxis(s, 0, 2)
    return s








