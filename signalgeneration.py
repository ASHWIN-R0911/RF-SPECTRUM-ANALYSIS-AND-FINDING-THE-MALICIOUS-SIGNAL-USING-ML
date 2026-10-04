# =========================================
# RF Signal Generation and Mixing
# Project: Malicious RF Signal Detection
# =========================================
import os
os.chdir(r'E:\D\projectsml')
import numpy as np
import matplotlib.pyplot as plt
import csv

# =========================================
# PARAMETERS
# =========================================
fs          = 2.048e6       # Sampling frequency: 2.048 MHz
duration    = 0.05          # Signal duration: 50 milliseconds
t           = np.arange(0, duration, 1/fs)
fc          = 433.92e6      # Normal signal carrier: 433.92 MHz (ISM band)
fc_jammer   = 433.92e6      # Jammer frequency: 433.92 MHz (same ISM band)
JSR_dB      = 15            # Jammer-to-Signal Ratio: +15 dB
SNR_dB      = 12            # Signal-to-Noise Ratio: +12 dB
bit_rate    = 1e3           # Bit rate: 1 kbps (OOK standard)

print("=" * 50)
print("RF Signal Generation Started")
print("=" * 50)
print(f"Sampling Rate      : {fs/1e6:.3f} MHz")
print(f"Signal Duration    : {duration*1e3:.0f} ms")
print(f"Normal Signal Freq : {fc/1e6:.2f} MHz")
print(f"Jammer Frequency   : {fc_jammer/1e6:.2f} MHz")
print(f"JSR                : {JSR_dB} dB")
print(f"SNR                : {SNR_dB} dB")
print("=" * 50)

# =========================================
# STEP 1: GENERATE NORMAL / BENIGN SIGNAL
# =========================================
samples_per_bit = int(fs / bit_rate)
num_bits        = int(duration * bit_rate)

np.random.seed(42)
bits         = np.random.randint(0, 2, num_bits)
ook_envelope = np.repeat(bits, samples_per_bit)
ook_envelope = ook_envelope[:len(t)]

# OOK modulation onto baseband carrier
carrier       = np.cos(2 * np.pi * (fs * 0.1) * t)
normal_signal = ook_envelope * carrier

# Add realistic background noise
signal_power  = np.mean(normal_signal[normal_signal != 0] ** 2)
noise_power   = signal_power / (10 ** (SNR_dB / 10))
noise         = np.sqrt(noise_power) * np.random.randn(len(t))
normal_signal = normal_signal + noise

print("Step 1 DONE: Normal OOK signal generated")

# =========================================
# STEP 2: GENERATE MALICIOUS SIGNAL
# =========================================
# Tone jammer: pure sine wave at same ISM band
tone_jammer = np.sin(2 * np.pi * (fs * 0.1) * t)

# Wideband noise jammer component
noise_jammer = np.random.randn(len(t))

# Combined malicious signal: 70% tone + 30% noise jamming
malicious_signal = 0.7 * tone_jammer + 0.3 * noise_jammer

print("Step 2 DONE: Malicious signal (tone + noise jammer) generated")

# =========================================
# STEP 3: MIX NORMAL + MALICIOUS SIGNAL
# =========================================
JSR_linear    = 10 ** (JSR_dB / 20)
normal_rms    = np.sqrt(np.mean(normal_signal ** 2))
malicious_rms = np.sqrt(np.mean(malicious_signal ** 2))
scale_factor  = JSR_linear * (normal_rms / malicious_rms)

mixed_signal = normal_signal + scale_factor * malicious_signal

print("Step 3 DONE: Signals mixed at JSR =", JSR_dB, "dB")

# =========================================
# STEP 4: COMPUTE FFT OF MIXED SIGNAL
# =========================================
N             = len(mixed_signal)
fft_result    = np.fft.fft(mixed_signal)
fft_magnitude = np.abs(fft_result[:N//2])
fft_magnitude_dB = 20 * np.log10(fft_magnitude + 1e-12)
frequencies   = np.fft.fftfreq(N, 1/fs)[:N//2]

print("Step 4 DONE: FFT computed")

# =========================================
# STEP 5: PLOT ALL SIGNALS
# =========================================
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle('RF Malicious Signal Detection - Signal Generation',
             fontsize=14, fontweight='bold')

# Plot 1: Normal signal
axes[0].plot(t * 1e3, normal_signal,
             color='blue', linewidth=0.6, label='Normal OOK Signal')
axes[0].set_title('Step 1: Normal / Benign Signal (OOK Modulated | 433.92 MHz ISM Band)')
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Amplitude')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Plot 2: Malicious signal
axes[1].plot(t * 1e3, malicious_signal,
             color='red', linewidth=0.6, label='Malicious Signal (Jammer)')
axes[1].set_title('Step 2: Malicious Signal (Tone + Noise Jammer | 433.92 MHz ISM Band)')
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Amplitude')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# Plot 3: Mixed signal
axes[2].plot(t * 1e3, mixed_signal,
             color='purple', linewidth=0.6, label='Mixed Signal')
axes[2].set_title(f'Step 3: Mixed Signal (Normal + Malicious | JSR = +{JSR_dB} dB)')
axes[2].set_xlabel('Time (ms)')
axes[2].set_ylabel('Amplitude')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

# Plot 4: FFT Spectrum
axes[3].plot(frequencies / 1e3, fft_magnitude_dB,
             color='black', linewidth=0.8, label='FFT Spectrum')
axes[3].set_title('Step 4: Frequency Spectrum of Mixed Signal (FFT) → CA-CFAR Input')
axes[3].set_xlabel('Frequency (kHz) — Baseband of 433.92 MHz ISM Band')
axes[3].set_ylabel('Power (dB)')
axes[3].legend(loc='upper right')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('signal_generation_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("Step 5 DONE: Plots saved as signal_generation_output.png")

# =========================================
# STEP 6: EXPORT MIXED SIGNAL TO CSV
# =========================================
with open('mixed_rf_signal.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Time_sec', 'Normal_Signal',
                     'Malicious_Signal', 'Mixed_Signal'])
    for i in range(len(t)):
        writer.writerow([
            round(t[i], 10),
            round(normal_signal[i], 8),
            round(malicious_signal[i], 8),
            round(mixed_signal[i], 8)
        ])

print("Step 6 DONE: mixed_rf_signal.csv saved")
print("=" * 50)
print("Signal generation complete!")
print("Next step: Run ca_cfar.py on the mixed signal")
print("=" * 50)