
import numpy as np

from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler((0, 255))
scaler2 = MinMaxScaler()


def reshape_signal0(signal):
    # try diffrent signal reshape verison here
    m = 0
    signal[0], signal[-1] = None, None
    for s in signal:
        if s is not None and s > m:
            m = s
    for i in range(len(signal)):
        if signal[i] is None:
            signal[i] = m
    return reshape_signal1(signal)


def reshape_signal1(signal):
    s = np.array([[float(i)] for i in signal])
    s = scaler.fit_transform(s)
    s = s.astype(int)
    s = s.reshape(36, 36)
    s = [s, s, s]
    s = np.moveaxis(s, 0, 2)
    return s


def reshape_signal2(s1):
    s2 = np.append(np.diff(s1), 0)
    s3 = np.append(np.diff(s2), 0)
    s = [placeholder(s1), placeholder(s2), placeholder(s3)]
    s = np.moveaxis(s, 0, 2)
    return s


def placeholder(s):
    s = np.array([[float(i)] for i in s])
    s = scaler.fit_transform(s)
    s = s.astype(int)
    s = s.reshape(36, 36)
    return s


def scale_signal(signal):
    s = np.array([[float(i)] for i in signal])
    s = scaler.fit_transform(s, )
    s = s.astype(int)
