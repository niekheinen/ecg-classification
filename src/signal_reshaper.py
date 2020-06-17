# import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import itertools
import collections

import wfdb


from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

# All relevant symbols for this research.
relsym = list("NLRejAaSJV!EF/fQ")
aami = ["Normal", "Supraventricular", "Ventricular", "Fusion beat", "Unkown"]

# Window size is 1.8s, mutiply by 360 because of the ECG signals frequency
window = int(1.8 * 360)
scaler = MinMaxScaler()
beats = []


def ba_num(ba):
    return relsym.index(ba)


def aami_num(ba):
    if ba in set("NLRej"):
        return 0
    if ba in set("AaSJ"):
        return 1
    if ba in set("V!E"):
        return 2
    if ba == "F":
        return 3
    if ba in set("/fQ"):
        return 4
    else:
        return 5


class Beat:
    def __init__(self, ba, patient):
        self.ba = ba
        self.aami_num = aami_num(ba)
        self.patient = int(patient)


def format_signal(signal, mid, w=window, extend_signal=True):
    # left, right = [], [] not necessary
    # sel and ser are Signal Extension Left and Right.
    if extend_signal is None:
        sel, ser = None, None
    if extend_signal:
        sel, ser = signal[0], signal[-1]
    else:
        sel, ser = 0, 0
    start = mid - w
    if start >= 0:
        left = signal[start:mid]
    else:
        left = [sel] * (start * -1)
        left.extend(signal[:mid])
    right = signal[mid:mid + w]
    if len(right) != w:
        n = [ser] * (w - len(right))
        right.extend(n)
    left.extend(right)
    assert len(left) == w * 2
    return left


def reshape_signal(signal):
    s = np.array([[float(i)] for i in signal])
    s = scaler.fit_transform(s)
    s = np.array([[int(i * 255)] * 3 for i in s])
    return s.reshape(36, 36, 3)






