import numpy as np

from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter

scaler = MinMaxScaler((0, 255))


# Takes a signal and transforms it into a tensor
def to_tensor(signal):
    red = signal
    savgol = savgol_filter(signal, 25, 2)
    green = savgol_filter([abs(i) for i in (signal - savgol)], 101, 2)
    blue = savgol_filter(green, 101, 2)
    green = set_limits(green, 0.01, 0.05)
    tensor = [to_channel(red), to_channel(green), to_channel(blue)]
    return np.moveaxis(tensor, 0, 2)


# Takes a signal and transforms it into a tensor channel
def to_channel(signal):
    s = np.array([[float(i)] for i in signal])
    s = scaler.fit_transform(s)
    s = s.astype(int)
    s = s.reshape(32, 32)
    return s


# Rather than scaling the entire signal based on min(signal) and max(signal), use this function to set limits
def set_limits(signal, y1, y2):
    assert y1 < y2, 'y1 has to be smaller than y2'
    res = []
    for s in signal:
        if s < y1:
            res.append(y1)
        elif s > y2:
            res.append(y2)
        else:
            res.append(s)
    return res
