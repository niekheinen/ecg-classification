from signal import *
import datetime

# Patients used for training, validation and test set to evaluate the final model. V1 is how
trainV1 = ['103', '105', '107', '108', '111', '112', '114', '115', '117', '119', '121', '123',
           '124', '200', '202', '205', '208', '209', '210', '212', '214', '215', '217', '219',
           '222', '223', '230', '231', '233']
validV1 = ['101', '106', '109', '113', '122', '201', '203', '213', '221', '228', '232', '234']
testsV1 = ['100', '116', '118', '207', '220']

# DS1 and DS2 from Rahal2016
train = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
valid = []
tests = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


def load_data(filepath):
    measuretime = datetime.now()
    for file in os.listdir(filepath):
        if file.endswith(".atr"):
            record = wfdb.rdrecord(filepath + file.split(".")[0])
            ann = wfdb.rdann(filepath + record.record_name, 'atr')
            end = 0
            for i in range(ann.ann_len - 3):
                ba = ann.symbol[i]
                if ba in relsym:
                    beat = Beat(ba, record.record_name)
                    beat.start = end
                    end = int((ann.sample[i] + ann.sample[i + 1]) / 2)
                    beat.end = end
                    signal = [t[0] for t in record.p_signal[beat.start:beat.end]]
                    beat.signal = format_signal(signal, ann.sample[i] - beat.start)
                    beat.mid = ann.sample[i] - beat.start
                    # beat.data = reshape_signal(beat.signal)
                    beats.append(beat)

    print("Loading the data took: {}s".format(round((datetime.now() - measuretime).total_seconds(), 1)))
    measuretime = datetime.now()
    data = {'train': ([], []), 'valid': ([], []), 'tests': ([], [])}
    for b in beats:
        data = reshape_signal(b.signal)
        label = ba_num(b.ba)
        if b.patient in train:
            s = 'train'
        elif b.patient in valid:
            s = 'valid'
        elif b.patient in tests:
            s = 'tests'
        else:
            raise Exception("Patient not assigned to a dataset")
        data[s][0].append(data)
        data[s][1].append(label)
    for k in data:
        data[k][0] = np.array([k][0])
        data[k][1] = np.array([k][1])
    print("Reshaping the signals took: {}s".format(round((datetime.now() - measuretime).total_seconds(), 1)))
    return data
