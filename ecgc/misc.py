from matplotlib import pyplot as plt


# Creates a schematic representation of an ECG signal. Figure 1 of my paper.
def make_figure():
    wave = [0] * 80
    wave.extend(parabola(80, 0.1))
    wave.extend([0] * 50)
    wave.extend(linear(15, -0.15))
    wave.extend(linear(20, 1.15, -0.15))
    wave.extend(linear(20, -1.3, 1))
    wave.extend(linear(20, 0.3, - 0.3))
    wave.extend([0] * 50)
    wave.extend(parabola(80, 0.17))
    wave.extend([0] * 80)
    plt.plot(wave, linewidth=5, color='k')
    plt.text(110, 0.15, 'P', fontsize=18, fontweight='bold')
    plt.text(215, -0.27, 'Q', fontsize=18, fontweight='bold')
    plt.text(235, 1.05, 'R', fontsize=18, fontweight='bold')
    plt.text(255, -0.42, 'S', fontsize=18, fontweight='bold')
    plt.text(365, 0.22, 'T', fontsize=18, fontweight='bold')
    pwave = [None] * 80
    pwave.extend([-0.1] * 80)
    pwave[-80], pwave[-1] = -0.05, -0.05
    qrs = [None] * 220
    qrs.extend([1.2] * 50)
    qrs[-50], qrs[-1] = 0.6, 0.6
    twave = [None] * 335
    twave.extend([-0.1] * 80)
    twave[-80], twave[-1] = -0.05, -0.05
    pqrst = [None] * 80
    pqrst.extend([-0.48] * 335)
    pqrst[80], pqrst[-1] = -0.25, -0.25
    plt.plot(pwave, linewidth=3, color='tab:blue')
    plt.plot(qrs, linewidth=3, color='tab:red')
    plt.plot(twave, linewidth=3, color='tab:orange')
    plt.plot(pqrst, linewidth=3, color='tab:green')
    plt.text(90, -0.18, 'P-wave', fontsize=12)
    plt.text(190, 1.25, 'QRS-complex', fontsize=12)
    plt.text(345, -0.18, 'T-wave', fontsize=12)
    plt.text(200, -0.58, 'PQRST-wave', fontsize=12)
    plt.ylim(-0.65, 1.4)
    plt.xlim(right=580)
    plt.show()


def linear(length, height, b=0):
    a = height / length
    return [a * x + b for x in range(length)]


def parabola(length, height):
    a = (height / ((length / 2) ** 2))
    return [-1 * (a * (x - (length / 2)) ** 2) + height for x in range(length)]


if __name__ == '__main__':
    make_figure()
