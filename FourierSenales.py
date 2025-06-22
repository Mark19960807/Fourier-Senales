import numpy as np
import matplotlib.pyplot as plt

# Parámetros comunes
t = np.linspace(-1, 1, 1000, endpoint=False)
fs = 1000
N = len(t)

# Lista para las señales
signals = [
    ("Pulso rectangular", np.where((t >= -0.2) & (t <= 0.2), 1.0, 0.0)),
    ("Función escalón", np.where(t >= 0, 1.0, 0.0)),
    ("Función senoidal", np.sin(2 * np.pi * 5 * t)),
]

fig, axes = plt.subplots(len(signals), 3, figsize=(15, 10))
fig.suptitle("Señales y su Transformada de Fourier", fontsize=16)

for i, (title, signal) in enumerate(signals):
    # FFT
    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(N, 1/fs)
    magnitude = np.abs(fft_vals) / N
    phase = np.angle(fft_vals)

    # Dominio del tiempo
    axes[i, 0].plot(t, signal)
    axes[i, 0].set_title(f"{title} (Tiempo)")
    axes[i, 0].set_xlabel("t [s]")

    # Magnitud
    axes[i, 1].plot(fft_freqs, magnitude)
    axes[i, 1].set_title("Magnitud FFT")
    axes[i, 1].set_xlim(-50, 50)

    # Fase
    axes[i, 2].plot(fft_freqs, phase)
    axes[i, 2].set_title("Fase FFT")
    axes[i, 2].set_xlim(-50, 50)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
