
noSmp       = 4096         # number of sampling points
bw          = 8000         # [Hz] sampling frequency of the ADCs
ppmRange    = (-2, 7)      # [ppm] range where to plot spectrum

kwargs_BS = {'bw':          bw,
             'noSmp':       noSmp,
             'lWidth':      (1,5),           # [Hz] Lorentz Width (MMBG: [10 20]; Metab: [2.0 8.0])
             'gWidth':      (1,6),           # [Hz] Gauss Width (equal for MMBG and Metab, default [1.5 5.0])
             'phaseOffs':   (40-25,40+25),   # [Deg] phase variation
             'freq_offset': (-20,20),        # [Hz] frequency offset
             'noiseLvl':    0.05             # noise level
             }

kwargs_MM = {'bw':          bw,
             'noSmp':       noSmp,
             'globalAmp':   2*7.9/1.3,
             'sdGlobalAmp': 0.1,
             'sdGlobalL':   3,
             'sdMMAmp':     0.05,
             'sdPhase':     20,
             'sdFreq':      0.0,
             }