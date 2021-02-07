import numpy as np
import os
import wfdb
from datetime import datetime
import signal_reshaper

# Window size is 1.8s, mutiply by 360 because of the ECG signals frequency
window = int(1.8 * 360)

# No longer used, keeping it a bit longer just to be sure.
# # Patients used for training, validation and test set to evaluate the final model. V1 is how
# train = ['103', '105', '107', '108', '111', '112', '114', '115', '117', '119', '121', '123',
#            '124', '200', '202', '205', '208', '209', '210', '212', '214', '215', '217', '219',
#            '222', '223', '230', '231', '233']
# valid = ['101', '106', '109', '113', '122', '201', '203', '213', '221', '228', '232', '234']
# tests = ['100', '116', '118', '207', '220']

# DS1 and DS2 from Rahal2016
train = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
valid = []
tests = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# All relevant symbols for this research.
relsym = list('NLRejAaSJV!EF/fQ')


def ba_num(ba):
    return relsym.index(ba)


def aami_num(ba):
    if ba in set('NLRej'):
        return 0
    if ba in set('AaSJ'):
        return 1
    if ba in set('V!E'):
        return 2
    if ba == 'F':
        return 3
    if ba in set('/fQ'):
        return 4
    else:
        return 5


class Beat:
    def __init__(self, ba, patient):
        self.ba = ba
        self.aami_num = aami_num(ba)
        self.patient = int(patient)


def format_signal(signal, mid, extend_signal=True, w=window):
    # sel and ser are Signal Extension Left and Right.
    if extend_signal is None:
        sel, ser = None, None
    elif extend_signal:
        sel, ser = signal[0], signal[-1]
    else:
        sel, ser = min(signal), min(signal)
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


def load_beats(filepath, patients=[], signal_format='ES'):
    beats, measuretime, end = [], datetime.now(), 0
    for file in os.listdir(filepath):
        if file.endswith('.atr'):
            record = wfdb.rdrecord(filepath + file.split('.')[0])
            if not patients or int(record.record_name) in patients:
                ann = wfdb.rdann(filepath + record.record_name, 'atr')
                for i in range(ann.ann_len - 3):
                    ba = ann.symbol[i]
                    if ba in relsym:
                        beat = Beat(ba, record.record_name)
                        beat.start = end
                        end = int((ann.sample[i] + ann.sample[i + 1]) / 2)
                        beat.end = end
                        if signal_format == 'OS':
                            signal = [t[0] for t in record.p_signal[ann.sample[i]-window:ann.sample[i]+window]]
                        signal = [t[0] for t in record.p_signal[beat.start:beat.end]]
                        if signal:
                            if signal_format == 'ES':
                                beat.signal = format_signal(signal, ann.sample[i] - beat.start, extend_signal=True)
                            elif signal_format == 'IS':
                                beat.signal = format_signal(signal, ann.sample[i] - beat.start, extend_signal=None)
                            elif signal_format == 'MS':
                                beat.signal = format_signal(signal, ann.sample[i] - beat.start, extend_signal=False)
                            beat.mid = ann.sample[i] - beat.start
                            beats.append(beat)
    print('Loading beats took: {}s'.format(round((datetime.now() - measuretime).total_seconds(), 1)))
    return beats


def load_beats_naive(filepath, patients=[]):
    beats, end = [], 0
    for file in os.listdir(filepath):
        if file.endswith('.atr'):
            record = wfdb.rdrecord(filepath + file.split('.')[0])
            if not patients or int(record.record_name) in patients:
                ann = wfdb.rdann(filepath + record.record_name, 'atr')
                for i in range(ann.ann_len - 3):
                    ba = ann.symbol[i]
                    if ba in relsym:
                        beat = Beat(ba, record.record_name)
                        mid = ann.sample[i]
                        signal = [t[0] for t in record.p_signal[mid-window:mid+window]]
                        if signal:
                            beat.signal = np.asarray(signal)
                            beat.start = mid - window
                            beats.append(beat)
    return beats


def load_data(filepath, classes='aami', patients=[], signal_format='ES'):
    if signal_format == 'OS':
        beats = load_beats_naive(filepath, patients)
    else:
        beats = load_beats(filepath, patients=patients, signal_format=signal_format)
    measuretime = datetime.now()
    data = {'train': [[], []], 'valid': [[], []], 'tests': [[], []]}
    for b in beats:
        if b.patient in train:
            s = 'train'
        elif b.patient in valid:
            s = 'valid'
        elif b.patient in tests:
            s = 'tests'
        else:
            continue
        if signal_format == 'IS':
            tensor = signal_reshaper.reshape_signal0(b.signal)
        else:
            print(b.signal.shape)
            tensor = signal_reshaper.reshape_signal1(b.signal)
        if classes == 'aami':
            label = b.aami_num
        else:
            label = ba_num(b.ba)
        data[s][0].append(tensor)
        data[s][1].append(label)
    for k in data:
        data[k][0] = np.asarray(data[k][0])
        data[k][1] = np.array(data[k][1])
    print('Reshaping the signals took: {}s'.format(round((datetime.now() - measuretime).total_seconds(), 1)))
    return data



