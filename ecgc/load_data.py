import numpy as np
import os
import wfdb
from scipy.signal import resample
from datetime import datetime

import ecgc.signal_reshaper as sr
import ecgc.config as config


# Returns number representation of BA class
def ba_num(ba):
    return config.relsym.index(ba)


# Returns number representation of AAMI class
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


# Heartbeat class
class Beat:
    def __init__(self, ba, patient):
        self.ba = ba
        self.aami_num = aami_num(ba)
        self.patient = int(patient)


def format_signal(signal, mid, extend_signal=True, half_window=config.hw):
    if not signal:
        return None
    if extend_signal is None:
        sel, ser = None, None
    elif extend_signal:
        sel, ser = signal[0], signal[-1]
    else:
        sel, ser = min(signal), min(signal)
    start = mid - half_window
    if start >= 0:
        left = signal[start:mid]
    else:
        left = [sel] * (start * -1)
        left.extend(signal[:mid])
    right = signal[mid:mid + half_window]
    if len(right) != half_window:
        n = [ser] * (half_window - len(right))
        right.extend(n)
    left.extend(right)
    if len(left) == half_window * 2:
        return left
    return None


def load_beats(filepath, patients=[], signal_format='ES'):
    beats, measuretime, end = [], datetime.now(), 0
    for file in os.listdir(filepath):
        if file.endswith('.atr'):
            record = wfdb.rdrecord(filepath + file.split('.')[0])

            # Skip if patient is not in specified patients.
            if patients and not int(record.record_name) in patients:
                continue

            ann = wfdb.rdann(filepath + record.record_name, 'atr')
            signal = np.array([i[0] for i in record.p_signal])
            signal, ann_sample = resample(signal, int(config.signal_frequency / 360) * len(signal), ann.sample)

            for i in range(ann.ann_len - 3):
                ba = ann.symbol[i]
                if ba in config.relsym:
                    beat = Beat(ba, record.record_name)
                    beat.start = end
                    end = int((ann.sample[i] + ann.sample[i + 1]) / 2)
                    beat.end = end
                    if signal_format == 'OS':
                        beat.signal = signal[ann.sample[i] - config.hw:ann.sample[i] + config.hw]
                    else:
                        beat_signal = [t[0] for t in record.p_signal[beat.start:beat.end]]
                        beat.mid = ann.sample[i] - beat.start
                        if signal_format == 'ES':
                            extend_signal = True
                        elif signal_format == 'IS':
                            extend_signal = None
                        elif signal_format == 'MS':
                            extend_signal = False
                        beat.signal = format_signal(beat_signal, beat.mid, extend_signal=extend_signal)
                    if beat.signal:
                        beats.append(beat)
    print('Loading beats took: {}s'.format(round((datetime.now() - measuretime).total_seconds(), 1)))
    return beats


def load_data(filepath, classes='aami', patients=[], signal_format='ES'):
    if signal_format == 'OS':
        pass
        # beats = load_beats_naive(filepath, patients)
    else:
        beats = load_beats(filepath, patients=patients, signal_format=signal_format)
    measuretime = datetime.now()
    data = {'train': [[], []], 'valid': [[], []], 'tests': [[], []]}
    for b in beats:
        if b.patient in config.train:
            s = 'train'
        elif b.patient in config.valid:
            s = 'valid'
        elif b.patient in config.tests:
            s = 'tests'
        else:
            continue
        if signal_format == 'IS':
            tensor = sr.reshape_signal0(b.signal)
        else:
            print(b.signal.shape)
            tensor = sr.reshape_signal1(b.signal)
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
