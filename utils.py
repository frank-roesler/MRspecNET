import numpy as np
import torch
from numpy.random import rand, randn
from scipy.fft import ifft, fftshift
import scipy.io
from glob import glob
import matplotlib.pyplot as plt


def plot_losses(losses, y_mode='log'):
    """plots the loss curve, together with its moving average during NN training"""
    t = range(len(losses))
    window_size = 50
    window = np.ones(window_size) / window_size
    losses_smooth = np.convolve(losses, window, mode='valid')

    tt = range(window_size // 2, len(losses_smooth) + window_size // 2)
    plt.cla()
    if y_mode == 'log':
        plt.semilogy(t, losses, linewidth=0.5)
        plt.semilogy(tt, losses_smooth, linewidth=1.5)
    else:
        plt.plot(t, losses, tt, losses_smooth, linewidth=0.5)
    plt.title('Loss')
    plt.show(block=False)
    plt.pause(0.01)


def print_info(losses, lr, epoch, epochs, n_params):
    print('Number of model parameters: ', n_params)
    print('Learning rate:', lr)
    print('-' * 50)
    print('Training. Epoch: ', str(epoch) + '/' + str(epochs), ', ', 'Loss: ', "{:e}".format(np.mean(losses[-100:])))
    print('-' * 50)


def build_ppmAx(bw, noSmp):
    gamma = 42.577  # [MHz/T]
    Bo = 2.89  # [T]
    wCenter = 4.65  # [ppm] Center frequency of Water
    fL = Bo * gamma
    fAx = np.arange(-bw / 2 + bw / (2 * noSmp), (bw / 2) - bw / (2 * noSmp), bw / (noSmp + 1))  # [Hz]
    ppm = fAx / fL
    ppm = ppm + wCenter
    return ppm, fAx, wCenter, fL


def voigtFunc(para, tAx):
    constL = np.pi
    constG = 2 * np.pi / np.sqrt(16 * np.log(2))
    osc = para['amplitude'] * np.exp(-1j*np.deg2rad(para['freq_offset']*360)*tAx + 1j*np.deg2rad(para['phase_offset']))
    damp = np.exp(-(constL * para['lorentz_width']) * tAx - (constG * para['gauss_width']) ** 2 * tAx ** 2)
    return osc * damp


def metabFunc(para, tAx, concentrations, errors, y_metab):
    """Adds Gauss and Lorentz damping, frequency and phase shift to the signal y_metab for the metabolite metab_name"""
    c = np.array(list(concentrations.values()))
    e = np.array(list(errors.values()))
    amplitudes = c + e*randn(*e.shape)
    constL     = np.pi
    constG     = 2 * np.pi / np.sqrt(16 * np.log(2))
    osc        = amplitudes * y_metab
    phase      = np.exp(1j*np.deg2rad(para['phaseOffs']))
    freq_shift = np.exp(-1j*np.deg2rad(para['freq_offset']*360)*tAx[:,np.newaxis])
    damp       = np.exp(-(constL*para['lorentz_width'])*tAx[:,np.newaxis] - (constG*para['gauss_width']*tAx[:,np.newaxis])**2)
    y = osc * damp * phase * freq_shift

    return y


def add_noise(signal, noiseLvl):
    noise = (noiseLvl * rand()) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    noisy_signal = signal + noise
    return noisy_signal


def add_mmbg(signal, noSmp, bw, globalPara, mmPara, globalAmp=1, sdGlobalAmp=0.1, sdGlobalL=3, sdMMAmp=0.05, sdPhase=20, sdFreq=0.2):
    """adds macromolecular baseline to signal.
    Input:  sdGlobalAmp in %, sdMMAmp in %, sdPhase in °, sdFreq in ppm, tAx in s, signal in time domain
    Output: signal with MMBG in time domain"""
    tAx = np.arange(noSmp) / bw
    gamma = 42.577  # [MHz/T]
    Bo = 2.89  # [T]
    fL = Bo * gamma  # [MHz]

    gblPara = {}
    gblPara['amplitude'] = globalAmp + sdGlobalAmp * globalAmp * randn()
    gblPara['freq_offset'] = globalPara['freq_offset'] + (sdFreq) * fL * randn()
    gblPara['phase_offset'] = sdPhase * randn()
    gblPara['lorentz_width'] = globalPara['lorentz_width'] + sdGlobalL*randn()
    gblPara['gauss_width'] = globalPara['gauss_width']

    paraVoigt = {}
    paraVoigt['phase_offset'] = 0
    paraVoigt['lorentz_width'] = 0
    paraVoigt['gauss_width'] = 0
    paraVoigt['amplitude']   = mmPara[:,0] + sdMMAmp * mmPara[:,0] * randn(mmPara.shape[0])
    paraVoigt['amplitude']   = paraVoigt['amplitude'][:, np.newaxis]
    paraVoigt['freq_offset'] = mmPara[:,1]
    paraVoigt['freq_offset'] = paraVoigt['freq_offset'][:, np.newaxis]

    signalOut = np.sum(voigtFunc(paraVoigt, tAx), axis=0)
    signalOut = signalOut * voigtFunc(gblPara, tAx)
    signalOut = signalOut.squeeze() + signal
    return signalOut


def make_batch(batch_size, simulator, device, include_mmbg=True, restrict_range=None, normalization=1, **kwargs_BS):
    batch = []
    for i in range(batch_size):
        signal = simulator.simulate(mmbg=include_mmbg, normalization=normalization)
        noisy_signal = add_noise(signal, kwargs_BS['noiseLvl'])
        if restrict_range:
            a,b = restrict_range
            signal = signal[a:b]
            noisy_signal = noisy_signal[a:b]
        signal = torch.from_numpy(signal).cfloat()
        signal = torch.stack([torch.real(signal),torch.imag(signal)])
        noisy_signal = torch.from_numpy(noisy_signal).cfloat()
        noisy_signal = torch.stack([torch.real(noisy_signal),torch.imag(noisy_signal)])
        batch.append((signal, noisy_signal))
    signal_batch = torch.stack([s for (s,n) in batch]).to(device)
    noisy_batch  = torch.stack([n for (s,n) in batch]).to(device)
    return signal_batch, noisy_batch


class Metab_basis:
    def __init__(self, path, kwargs_BS):
        self.kwargs = kwargs_BS
        self.metab_paths = sorted(glob(path+'/*.mat'), key = lambda s: s.casefold())
        metab_names = [metab_path.split('/')[-1] for metab_path in self.metab_paths]
        metab_names = sorted([name.split('.')[0] for name in metab_names], key = lambda s: s.casefold())
        concentration_values = [0, 0, 0, 2.83, 0, 6.26, 0, 2.12, 1.77, 3.96, 10.99, 0.47, 0.08,
                                1.43, 0.51, 6.11, 9.81, 2.09, 0.92, 1.84, 1.92, 0.33, 1.70]
        error_values         = [0, 0, 0, 0.32, 0, 0.48, 0, 0.42, 0.51, 0.46, 0.55, 0.18, 0.07,
                                0.13, 0.20, 0.63, 0.40, 0.39, 0.11, 0.48, 0.51, 0.06, 0.24]
        self.metab_names = metab_names
        self.concentrations = dict(zip(metab_names, concentration_values))
        self.errors         = dict(zip(metab_names, error_values))

    def make_patterns(self):
        t = np.arange(self.kwargs['noSmp'])/self.kwargs['bw']
        naked_patterns = np.zeros((self.kwargs['noSmp'],len(self.concentrations.values())), dtype=np.complex128)
        ctr=0
        for matPath, name in zip(self.metab_paths, self.metab_names):
            mat = scipy.io.loadmat(matPath)
            mrsSet = mat['exptDat'][0][0][3].squeeze() * np.exp(np.pi * t)
            naked_patterns[:,ctr] = mrsSet
            ctr+=1
        return naked_patterns


class MMBG_basis:
    def __init__(self, mmbg_path, kwargs_MM):
        self.adcNAA = 1e-4  # mm²/s
        self.bRef = 200  # s/mm²
        matMMBG = scipy.io.loadmat(mmbg_path)
        MMBGpara = matMMBG['mmPara'].squeeze()
        MMBGpara_list = [[t[0].item(),t[1].item()] for t in MMBGpara]
        MMBGpara_arr = np.array(MMBGpara_list)
        globalPara = {}
        globalPara['freq_offset'] = matMMBG['globalPara'][0][0][1]
        globalPara['phase_offset'] = matMMBG['globalPara'][0][0][2]
        globalPara['lorentz_width'] = matMMBG['globalPara'][0][0][3]
        globalPara['gauss_width'] = matMMBG['globalPara'][0][0][4]

        self.globalPara = globalPara
        self.kwargs = {**kwargs_MM,
                     'globalPara': globalPara,
                     # 'mmPara': MMBGpara,
                     'mmPara': MMBGpara_arr,     # for vectorized version of add_mmbg
                     }


class Simulator:
    def __init__(self, metab_basis, mmbg_basis):
        self.metab_basis = metab_basis
        self.mmbg_basis  = mmbg_basis

    def add_mmbg(self, signal):
        # y = add_mmbg(signal, **self.mmbg_basis.kwargs)
        y = add_mmbg(signal, **self.mmbg_basis.kwargs)   # vectorized version
        return y

    def simulate(self, mmbg=True, normalization=None):
        """simulates MRS spectrum as linear combination of basis sets with varying widths"""
        t = np.arange(self.metab_basis.kwargs['noSmp']) / self.metab_basis.kwargs['bw']
        y = self.metab_basis.make_patterns()  # dimensions: (noSmp, #metabolites)
        para = {}
        para['gauss_width'] = self.metab_basis.kwargs['gWidth'][0] + (self.metab_basis.kwargs['gWidth'][1] - self.metab_basis.kwargs['gWidth'][0]) * rand()
        para['freq_offset'] = self.metab_basis.kwargs['freq_offset'][0] + (
                    self.metab_basis.kwargs['freq_offset'][1] - self.metab_basis.kwargs['freq_offset'][0]) * rand(y.shape[1])
        para['lorentz_width'] = self.metab_basis.kwargs['lWidth'][0] + (self.metab_basis.kwargs['lWidth'][1] - self.metab_basis.kwargs['lWidth'][0]) * rand(
            y.shape[1])
        para['phaseOffs'] = self.metab_basis.kwargs['phaseOffs'][0] + \
                            (self.metab_basis.kwargs['phaseOffs'][1] - self.metab_basis.kwargs['phaseOffs'][0])*rand()
        y = metabFunc(para, t, self.metab_basis.concentrations, self.metab_basis.errors, y)
        y = np.sum(y, 1)
        if mmbg:
            y = self.add_mmbg(y)
        y = fftshift(ifft(np.conj(y), axis=0), axes=0)

        if normalization=='max_1':
            y = y/np.max(np.abs(y))
        elif isinstance(normalization,float) or isinstance(normalization,int):
            y = y/normalization

        return y











