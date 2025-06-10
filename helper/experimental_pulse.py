import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def print_fit_params(opt, cov):
    for i in range(len(opt)):
        print("i:", opt[i], "+/-", np.sqrt(cov[i][i]))

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "wang_pulse_AB.dat")
data = np.loadtxt(data_path, skiprows=2)
time = data[:, 0]
A = data[:, 1]
B = data[:, 2]

N = len(time)
FFT_MIN = int(0.25 * N)
FFT_MAX = int(0.65 * N)

signal_A = hilbert(A)
env_A = np.abs(signal_A)

signal_B = hilbert(B)
env_B = np.abs(signal_B)

fig, axes = plt.subplots(nrows=3, sharex=True)

axes[0].plot(time, A, "-",label="$E$")
axes[0].plot(time, env_A, "--", label="Envelope")
axes[1].plot(time, B, "-",label="$E_B$")
axes[1].plot(time, env_B, "--", label="Envelope")

axes[2].plot(time, A + B)
axes[2].axvline(time[FFT_MIN], c='k')
axes[2].axvline(time[FFT_MAX], c='k')

tA = time[np.argmax(env_A)]
ampA = np.max(env_A)

tA = time[np.argmax(env_B)]
ampA = np.max(env_B)

axes[1].set_xlabel("Time (ps)")
axes[0].set_ylabel("$E_A$ (kV/cm)")
axes[1].set_ylabel("$E_B$ (kV/cm)")
axes[2].set_ylabel("$E_A + E_B$ (kV/cm)")
axes[0].legend()
fig.tight_layout()


from scipy.fft import rfft, rfftfreq
HBAR = 6.582119569509065698e-1 # meV ps
PAD = 8
fftA = np.abs(rfft(A, n=PAD*N))
fftB = np.abs(rfft(B, n=PAD*N))
fft_total = np.abs(rfft((A+B)[FFT_MIN:FFT_MAX], n=PAD*N))

dt = 0
for i in range(N - 1):
    dt += time[i + 1] - time[i]
dt /= N

freq = 2 * np.pi * HBAR * rfftfreq(PAD*N, dt)

fig2, ax2 = plt.subplots()
ax2.plot(freq, fftA / np.max(fftA), label="A")
ax2.plot(freq, fftB / np.max(fftB), label="B")
ax2.plot(freq, fft_total / np.max(fft_total), label="A+B")

peak_pos = freq[np.argmax(fft_total)]
ax2.axvline(peak_pos, ls="--", c="k")

# my analysis: 1.489 THz; Wang: 1.45 THz...?
# Answer: Depends on where we set the FFT frame
print(peak_pos)

ax2.set_xlabel(r"$\hbar \omega$ (meV)")
ax2.set_ylabel(r"FFT (normalized)")

plt.show()