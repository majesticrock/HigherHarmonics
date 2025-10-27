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
FFT_MIN = int(0.25 * N)
FFT_MAX = int(0.65 * N)

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
axes[2].axvline(time[FFT_MAX], c='k')

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
PAD = 8
fftA = rfft(A, n=PAD*N)
fftB = rfft(B, n=PAD*N)
fft_total = rfft((A+B), n=PAD*N)#[FFT_MIN:FFT_MAX]

dt = 0
for i in range(N - 1):
    dt += time[i + 1] - time[i]
dt /= N

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

def gaussian_filter(omega, omega_0, OMEGA_CUT):
    sigma = OMEGA_CUT / np.sqrt(-2 * np.log(0.01))
    return np.exp(-(omega - omega_0)**2 / (2 * sigma**2))

# Calculate sigma such that G(OMEGA_CUT) = 0.01
OMEGA_CUT = peak_pos  # or whatever cutoff you want

# Apply Gaussian filter centered at zero
fft_filtered = fft_total * gaussian_filter(freq, 0, OMEGA_CUT)

# Plot the filter shape on the FFT plot
ax2.plot(freq, gaussian_filter(freq, peak_pos, OMEGA_CUT), '--', color='gray', alpha=0.5, label='Gaussian filter')
ax2.axvline(OMEGA_CUT, ls=':', color='gray', alpha=0.5, label=r'$\omega_\mathrm{cut}$')

reconstructed = irfft(fft_filtered, n=PAD * N)

fig3, ax3 = plt.subplots()
ax3.plot(np.arange(N) * dt, reconstructed[:N], label=f"reconstructed", lw=1)
ax3.plot(time, (A + B), label="original (A + B)", alpha=0.8)
ax3.set_xlabel("Time (ps)")
ax3.set_ylabel("Field (kV/cm)")
ax3.legend()
ax3.set_title("Inverse FFT after Gaussian filtering")

plt.show()