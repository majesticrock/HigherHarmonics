import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

HBAR = 6.582119569509065698e-1 # meV ps

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
FFT_MIN = int(0. * N)
FFT_MAX = int(1 * N)

signal_A = hilbert(A)
env_A = np.abs(signal_A)

signal_B = hilbert(B)
env_B = np.abs(signal_B)

fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(6.8, 6))
fig.subplots_adjust(hspace=0.0)

def vector_potential(electric_field, time):
    dt = time[1] - time[0]
    return -np.cumsum(electric_field) * dt

def dirac_shift(vector_potential):
    # v_F [m/s] * hbar [meV ps] * e [C] / hbar [meV ps] * (A / c) [kV ps / cm]
    # = v_F [m/s] * e [C] * (A/c) [V s / m] * 10^{-7} = # [CV]
    # = v_F [m/s] * (A/c) [kV ps / cm] * 10^{-7} eV
    return 1.5e6 * 1e-7 * vector_potential

def momentum_shift(vector_potential):
    # d [m] * e [C] / hbar [meV ps] * (A/c) [kV ps / cm] = # [C kV / (cm meV)]
    # = d [m] * (A/c) [kV ps / cm] / hbar [meV ps] * 10^8
    return 1e-9 * 1e8 / HBAR * vector_potential

axes[0].plot(time, A, "-",label="$E$")
axes[0].plot(time, env_A, "--", label="Envelope")
#axes[0].plot(time, 100 * dirac_shift(vector_potential(A, time)), label=r"$\tilde{A}$ [eV / 100]")
axes[0].plot(time, 100 / np.pi * momentum_shift(vector_potential(A, time)), label=r"$100 \tilde{A} / \pi$")

axes[1].plot(time, B, "-",label="$E_B$")
axes[1].plot(time, env_B, "--", label="Envelope")
#axes[1].plot(time, 100 * dirac_shift(vector_potential(B, time)), label=r"$\tilde{A}$ [eV / 100]")
axes[1].plot(time, 100 / np.pi * momentum_shift(vector_potential(B, time)), label=r"$100\tilde{A} / \pi$")

axes[2].plot(time, A + B)
axes[2].axvline(time[FFT_MIN], c='k')
axes[2].axvline(time[FFT_MAX - 1], c='k')

tA = time[np.argmax(env_A)]
ampA = np.max(env_A)

tA = time[np.argmax(env_B)]
ampA = np.max(env_B)

axes[-1].set_xlabel("Time (ps)")
axes[0].set_ylabel("$E_A$ (kV/cm)")
axes[1].set_ylabel("$E_B$ (kV/cm)")
axes[2].set_ylabel("$E_A + E_B$ (kV/cm)")
axes[0].legend(loc="upper right")


from scipy.fft import rfft, rfftfreq, irfft
from scipy.ndimage import gaussian_filter1d
PAD = 1
fftA = rfft(A[FFT_MIN:FFT_MAX], n=PAD*N)
fftB = rfft(B[FFT_MIN:FFT_MAX], n=PAD*N)
fft_total = rfft((A+B)[FFT_MIN:FFT_MAX], n=PAD*N)

dt = np.average(np.diff(time))

freq = 2 * np.pi * HBAR * rfftfreq(PAD*N, dt)

fig2, ax2 = plt.subplots()
ax2.plot(freq, np.abs(fftA) / np.max(np.abs(fftA)), label="A")
ax2.plot(freq, np.abs(fftB) / np.max(np.abs(fftB)), label="B")
ax2.plot(freq, np.abs(fft_total) / np.max(np.abs(fft_total)), label="A+B")
ax2.legend()

peak_pos = freq[np.argmax(np.abs(fft_total))]
ax2.axvline(peak_pos, ls="--", c="k")

# my analysis: 1.489 THz; Wang: 1.45 THz...?
# Answer: Depends on where we set the FFT frame
print(peak_pos)

ax2.set_xlabel(r"$\hbar \omega$ (meV)")
ax2.set_ylabel(r"FFT (normalized)")
ax2.set_yscale('log')
ax2.set_xlim(0, 30)


fig3, axes3 = plt.subplots(nrows=3, sharex=True, figsize=(6.8, 6))
fig3.subplots_adjust(hspace=0.0)

fig4, axes4 = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(6.8, 6))
fig4.subplots_adjust(hspace=0.0)

cut = 2 * peak_pos

for ax, ax_fft, signal, fft_in in zip(axes3, axes4, [A, B, A+B], [fftA, fftB, fft_total]):
    ax.set_ylabel("Field (kV/cm)")
    ax_fft.set_ylabel("FFT")
    
    cut_f_signal = fft_in.copy()
    cut_f_signal[freq > cut] = 0
    # smooth magnitude of the FFT while preserving phase
    amp = np.abs(cut_f_signal)
    phase = np.angle(cut_f_signal)

    # smoothing width: use 10% of the main peak (converted to bins)
    df = freq[1] - freq[0]
    sigma_freq = max(peak_pos * 0.1, df)
    sigma_bins = max(1.0, sigma_freq / df)

    smooth_amp = gaussian_filter1d(amp, sigma_bins)

    # reconstruct complex spectrum and lightly attenuate very high frequencies
    cut_f_signal = smooth_amp * np.exp(1j * phase)
    cut_f_signal *= np.exp(-0.5 * (freq / cut)**2)

    filtered_pulse = irfft(cut_f_signal, N)

    ax.plot(time, filtered_pulse, label="Filtered")
    ax.plot(time, signal, label="Original", ls="--")
    
    N = len(filtered_pulse)
    filter_freq = 2 * np.pi * HBAR * rfftfreq(PAD*N, dt)
    filter_fft = rfft(filtered_pulse, n=PAD*N)
    
    ax_fft.plot(filter_freq, np.abs(filter_fft) / np.max(np.abs(filter_fft)), label="Filtered")
    ax_fft.plot(freq, np.abs(fft_in) / np.max(np.abs(fft_in)), label="Original", ls="--")
    ax_fft.set_ylabel(r"FFT (normalized)")
    
    print(1e-3 * filtered_pulse)
    
ax.legend()
axes3[-1].set_xlabel("Time (ps)")

axes4[-1].set_xlabel(r"$\hbar \omega$ (meV)")
axes4[0].set_yscale('log')
axes4[0].set_xlim(0, 30)

plt.show()