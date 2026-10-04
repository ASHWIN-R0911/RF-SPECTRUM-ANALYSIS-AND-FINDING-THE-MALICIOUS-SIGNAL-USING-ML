# =========================================
# CA-CFAR Detection — Fixed Version
# Project: Malicious RF Signal Detection
# Input  : mixed_rf_signal.csv
# Output : cfar_output.csv
# =========================================

import os
os.chdir(r'E:\D\projectsml')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

print("=" * 55)
print("  CA-CFAR Malicious Signal Detection Started")
print("=" * 55)

# =========================================
# STEP 1: LOAD MIXED SIGNAL FROM CSV
# =========================================
df           = pd.read_csv('mixed_rf_signal.csv')
mixed_signal = df['Mixed_Signal'].values
t            = df['Time_sec'].values
fs           = 2.048e6

print(f"  [LOAD] Signal loaded: {len(mixed_signal):,} samples")

# =========================================
# STEP 2: AVERAGED FFT (noise reduction)
# Split signal into 8 segments, average
# their FFT magnitudes to suppress noise
# =========================================
N          = len(mixed_signal)
num_avg    = 8
seg_len    = N // num_avg
fft_avg    = np.zeros(seg_len // 2)

for i in range(num_avg):
    seg       = mixed_signal[i*seg_len:(i+1)*seg_len]
    fft_seg   = np.abs(np.fft.fft(seg)[:seg_len//2])
    fft_avg  += fft_seg

fft_avg      /= num_avg
fft_db        = 20 * np.log10(fft_avg + 1e-12)
frequencies   = np.fft.fftfreq(seg_len, 1/fs)[:seg_len//2]
freq_res      = fs / seg_len

print(f"  [FFT]  Averaged FFT: {num_avg} segments x {seg_len:,} points")
print(f"         Frequency resolution: {freq_res:.2f} Hz/bin")

# =========================================
# STEP 3: CA-CFAR DETECTION
# =========================================
num_train = 64    # from 32
num_guard = 16    # from 8
pfa       = 1e-5  # from 1e-4
alpha     = num_train * (pfa ** (-1 / num_train) - 1)

print(f"\n  [CFAR] Parameters:")
print(f"         Training cells : {num_train} each side")
print(f"         Guard cells    : {num_guard} each side")
print(f"         Pfa            : {pfa}")
print(f"         Alpha          : {alpha:.4f}")

total_cells     = num_train + num_guard
cfar_threshold  = np.zeros(len(fft_db))
cfar_detections = np.zeros(len(fft_db), dtype=bool)

for i in range(total_cells, len(fft_db) - total_cells):
    left_train     = fft_db[i - total_cells : i - num_guard]
    right_train    = fft_db[i + num_guard + 1 : i + total_cells + 1]
    noise_est      = np.mean(np.concatenate([left_train, right_train]))
    cfar_threshold[i] = noise_est + alpha
    if fft_db[i] > cfar_threshold[i]:
        cfar_detections[i] = True

# =========================================
# STEP 4: POWER FILTER
# Only keep bins 25dB above median noise
# =========================================
noise_floor    = np.median(fft_db)
min_threshold  = noise_floor + 25
cfar_detections = cfar_detections & (fft_db > min_threshold)
detected_indices = np.where(cfar_detections)[0]

print(f"\n  [FILTER] Noise floor estimate : {noise_floor:.2f} dB")
print(f"           Min power threshold  : {min_threshold:.2f} dB")
print(f"           Bins after filter    : {len(detected_indices)}")

# =========================================
# STEP 5: FIND STRONGEST PEAK DIRECTLY
# Instead of clustering, find the single
# highest FFT bin as the malicious signal
# =========================================
# Find the global peak in FFT
peak_bin       = np.argmax(fft_db)
peak_power_db  = fft_db[peak_bin]
peak_freq_hz   = frequencies[peak_bin]

# Expand around peak to find bandwidth
# Walk left and right from peak until power
# drops 10dB below the peak
threshold_bw   = peak_power_db - 10
k_low  = peak_bin
k_high = peak_bin

while k_low > 0 and fft_db[k_low] > threshold_bw:
    k_low -= 1
while k_high < len(fft_db)-1 and fft_db[k_high] > threshold_bw:
    k_high += 1

# Calculate results
center_freq_hz = frequencies[k_low + (k_high - k_low) // 2]
bandwidth_hz   = (k_high - k_low) * freq_res

print(f"\n  [RESULT] Malicious Signal Detected!")
print(f"  {'─'*45}")
print(f"  Peak Frequency   : {peak_freq_hz/1e3:.4f} kHz (baseband)")
print(f"                     → 433.92 MHz ISM band (real-world)")
print(f"  Center Frequency : {center_freq_hz/1e3:.4f} kHz")
print(f"  Bandwidth        : {bandwidth_hz:.2f} Hz ({bandwidth_hz/1e3:.4f} kHz)")
print(f"  Peak Power       : {peak_power_db:.2f} dB")
print(f"  Noise Floor      : {noise_floor:.2f} dB")
print(f"  SNR (signal)     : {peak_power_db - noise_floor:.2f} dB above noise")
print(f"  Bin Range        : [{k_low} → {k_high}]")
print(f"  {'─'*45}")

# =========================================
# STEP 6: ISOLATE MALICIOUS WAVEFORM
# Band-pass via IFFT of detected region
# =========================================
# Use full FFT for isolation
fft_full      = np.fft.fft(mixed_signal)
fft_isolated  = np.zeros(N, dtype=complex)

# Scale bin indices back to full FFT size
scale         = N / seg_len
k_low_full    = int(k_low  * scale)
k_high_full   = int(k_high * scale)

# Copy only the jammer frequency region
fft_isolated[k_low_full  : k_high_full+1] = \
    fft_full[k_low_full  : k_high_full+1]
# Mirror for negative frequencies
fft_isolated[N-k_high_full : N-k_low_full+1] = \
    fft_full[N-k_high_full : N-k_low_full+1]

isolated_waveform = np.real(np.fft.ifft(fft_isolated))

print(f"  [ISO]  Waveform isolated: {len(isolated_waveform):,} samples")
print(f"         Waveform amplitude range: "
      f"[{isolated_waveform.min():.4f}, {isolated_waveform.max():.4f}]")

# =========================================
# STEP 7: VISUALIZATION — 3 PANELS
# =========================================
fig, axes = plt.subplots(3, 1, figsize=(14, 11))
fig.suptitle(
    'CA-CFAR Malicious Signal Detection\n'
    'Target: 433.92 MHz ISM Band  |  JSR: +15 dB  |  SNR: +12 dB',
    fontsize=13, fontweight='bold'
)

# --- Panel 1: FFT + CFAR Threshold ---
axes[0].plot(frequencies/1e3, fft_db,
             color='black', linewidth=0.6,
             label='FFT Power Spectrum (averaged)')
axes[0].plot(frequencies/1e3, cfar_threshold,
             color='orange', linewidth=1.5,
             linestyle='--', label='CA-CFAR Adaptive Threshold')
axes[0].axvline(x=peak_freq_hz/1e3, color='red',
                linewidth=1.5, linestyle='-',
                label=f'Jammer Peak: {peak_freq_hz/1e3:.2f} kHz')
axes[0].axhline(y=min_threshold, color='green',
                linewidth=1.2, linestyle=':',
                label=f'Power Filter: {min_threshold:.1f} dB')
axes[0].set_title('Panel 1: FFT Spectrum with CA-CFAR Adaptive Threshold')
axes[0].set_xlabel('Frequency (kHz)')
axes[0].set_ylabel('Power (dB)')
axes[0].legend(loc='upper right', fontsize=8)
axes[0].grid(True, alpha=0.3)

# --- Panel 2: Detection Map ---
axes[1].plot(frequencies/1e3, fft_db,
             color='black', linewidth=0.6,
             label='FFT Spectrum')
# Shade only the detected jammer bandwidth region
axes[1].axvspan(frequencies[k_low]/1e3, frequencies[k_high]/1e3,
                alpha=0.4, color='red',
                label=f'Malicious Region')
axes[1].axvline(x=center_freq_hz/1e3, color='red',
                linewidth=2.0, linestyle='-',
                label=f'Center: {center_freq_hz/1e3:.2f} kHz')
axes[1].axvline(x=frequencies[k_low]/1e3, color='green',
                linewidth=1.5, linestyle='--',
                label=f'BW Edges ({bandwidth_hz:.1f} Hz)')
axes[1].axvline(x=frequencies[k_high]/1e3, color='green',
                linewidth=1.5, linestyle='--')
axes[1].set_title(
    f'Panel 2: Detection Map  |  '
    f'Center: {center_freq_hz/1e3:.2f} kHz  |  '
    f'BW: {bandwidth_hz:.2f} Hz  |  '
    f'Peak: {peak_power_db:.2f} dB'
)
axes[1].set_xlabel('Frequency (kHz)')
axes[1].set_ylabel('Power (dB)')
axes[1].legend(loc='upper right', fontsize=8)
axes[1].grid(True, alpha=0.3)

# --- Panel 3: Isolated Malicious Waveform ---
# Plot only first 1000 samples for clarity
plot_samples = min(1000, len(isolated_waveform))
axes[2].plot(t[:plot_samples]*1e3,
             isolated_waveform[:plot_samples],
             color='red', linewidth=0.8,
             label='Isolated Malicious Waveform')
axes[2].set_title(
    f'Panel 3: Isolated Malicious Signal Waveform  |  '
    f'BW: {bandwidth_hz:.2f} Hz  |  '
    f'433.92 MHz ISM Band'
)
axes[2].set_xlabel('Time (ms)')
axes[2].set_ylabel('Amplitude')
axes[2].legend(loc='upper right', fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cfar_detection_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  [PLOT] Saved: cfar_detection_output.png")

# =========================================
# STEP 8: EXPORT RESULTS TO CSV
# =========================================
with open('cfar_output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'Center_Freq_kHz', 'Bandwidth_Hz',
        'Peak_Power_dB', 'Noise_Floor_dB',
        'SNR_above_noise_dB', 'Peak_Freq_kHz',
        'Bin_Low', 'Bin_High', 'Detection'
    ])
    writer.writerow([
        round(center_freq_hz/1e3, 4),
        round(bandwidth_hz, 4),
        round(peak_power_db, 4),
        round(noise_floor, 4),
        round(peak_power_db - noise_floor, 4),
        round(peak_freq_hz/1e3, 4),
        k_low, k_high,
        'MALICIOUS'
    ])

print("  [CSV]  Saved: cfar_output.csv")
print("\n" + "=" * 55)
print("  CA-CFAR Complete — Ready for feature_extraction.py")
print("=" * 55)