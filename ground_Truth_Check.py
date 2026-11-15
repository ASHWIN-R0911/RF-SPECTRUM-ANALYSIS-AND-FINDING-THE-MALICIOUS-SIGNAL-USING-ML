
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("mixed_rf_signal.csv")

# Extract malicious signal
m = df["Malicious_Signal"].values

# Calculate FFT power spectrum
P = np.abs(np.fft.rfft(m - m.mean())) ** 2

# Frequency axis
f = np.fft.rfftfreq(len(m), 1 / 2.048e6)

# Find strongest frequencies
top = np.argsort(P)[::-1][:5]

print("Strongest freqs (kHz):", np.round(f[top] / 1e3, 2))

# Spectral flatness
spectral_flatness = (
    np.exp(np.mean(np.log(P + 1e-12))) / np.mean(P)
)

print("Spectral flatness:", spectral_flatness)

# Envelope duty cycle
duty_cycle = np.mean(
    np.abs(m) > 0.5 * np.abs(m).max()
)

print("Envelope duty cycle:", duty_cycle)
