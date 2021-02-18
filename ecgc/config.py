# All variables used to configure the project

# Desired frequency of ECG signal in Hz
signal_frequency = 720

# With an ECG signal frequency of 720 Hz, this means the window is 1024/720 = 1.4222s
window = 1024

# Half window, used for loading signals
hw = int(window / 2)

# DS1 and DS2 from Rahal2016
train = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
valid = []
tests = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# All relevant symbols for this research.
relsym = list('NLRejAaSJV!EF/fQ')
