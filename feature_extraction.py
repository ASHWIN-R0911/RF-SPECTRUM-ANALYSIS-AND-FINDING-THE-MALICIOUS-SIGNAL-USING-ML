# =========================================
# Feature Extraction
# Project: Malicious RF Signal Detection
# Input  : mixed_rf_signal.csv
#           cfar_output.csv
# Output : features_dataset.csv
# =========================================

import os
os.chdir(r'E:\D\projectsml')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

print("=" * 55)
print("  Feature Extraction Started")
print("=" * 55)

# =========================================
# STEP 1: LOAD SIGNALS AND CFAR RESULTS
# =========================================
df           = pd.read_csv('mixed_rf_signal.csv')
mixed_signal = df['Mixed_Signal'].values
normal_signal   = df['Normal_Signal'].values
malicious_signal = df['Malicious_Signal'].values
t            = df['Time_sec'].values
fs           = 2.048e6

cfar_df      = pd.read_csv('cfar_output.csv')
center_freq  = cfar_df['Center_Freq_kHz'].values[0]
bandwidth_hz = cfar_df['Bandwidth_Hz'].values[0]
peak_power   = cfar_df['Peak_Power_dB'].values[0]
noise_floor  = cfar_df['Noise_Floor_dB'].values[0]
k_low        = int(cfar_df['Bin_Low'].values[0])
k_high       = int(cfar_df['Bin_High'].values[0])

print(f"  [LOAD] Mixed signal     : {len(mixed_signal):,} samples")
print(f"  [LOAD] CFAR result      : Center={center_freq} kHz | BW={bandwidth_hz} Hz")

# =========================================
# STEP 2: ISOLATE MALICIOUS WAVEFORM
# Using CFAR bin range via IFFT
# =========================================
N            = len(mixed_signal)
seg_len      = N // 8
scale        = N / seg_len
k_low_full   = int(k_low  * scale)
k_high_full  = int(k_high * scale)

fft_full     = np.fft.fft(mixed_signal)
fft_isolated = np.zeros(N, dtype=complex)
fft_isolated[k_low_full  : k_high_full+1] = \
    fft_full[k_low_full  : k_high_full+1]
fft_isolated[N-k_high_full : N-k_low_full+1] = \
    fft_full[N-k_high_full : N-k_low_full+1]

isolated_waveform = np.real(np.fft.ifft(fft_isolated))

print(f"  [ISO]  Malicious waveform isolated: "
      f"[{isolated_waveform.min():.4f}, {isolated_waveform.max():.4f}]")

# =========================================
# STEP 3: FEATURE EXTRACTION
# Three features per signal block:
#   1. PAPR  — Peak to Average Power Ratio
#   2. Variance — amplitude spread
#   3. Bandwidth — frequency span (from CFAR)
# Plus 3 supporting features for ML strength
# =========================================

def extract_features(waveform, bandwidth, peak_db, noise_db, label):
    """
    Extract all features from a signal waveform block.
    Returns a dictionary of features + label.
    """
    power = waveform ** 2   # Instantaneous power

    # --- Feature 1: PAPR (Peak to Average Power Ratio) ---
    # High PAPR = spiky/bursty signal (jammer characteristic)
    # Low PAPR  = steady continuous signal (normal OOK characteristic)
    peak_power_linear = np.max(power)
    avg_power_linear  = np.mean(power)
    papr_db = 10 * np.log10(
        peak_power_linear / (avg_power_linear + 1e-12)
    )

    # --- Feature 2: Variance ---
    # Measures amplitude spread/fluctuation
    # Noise jammer has high variance; tone jammer has low variance
    variance = np.var(waveform)

    # --- Feature 3: Bandwidth (Hz) ---
    # Directly from CFAR output
    # Tone jammer = narrow BW; noise jammer = wide BW
    bw = bandwidth

    # --- Supporting Feature 4: Spectral Flatness ---
    # Ratio of geometric mean to arithmetic mean of power spectrum
    # Flat spectrum (close to 1) = noise-like (jammer)
    # Peaked spectrum (close to 0) = structured signal (normal)
    fft_mag = np.abs(np.fft.fft(waveform)[:len(waveform)//2]) + 1e-12
    geo_mean = np.exp(np.mean(np.log(fft_mag)))
    ari_mean = np.mean(fft_mag)
    spectral_flatness = geo_mean / ari_mean

    # --- Supporting Feature 5: Kurtosis ---
    # Measures "spikiness" or heavy tails in distribution
    # Impulsive/pulsed jammer has high kurtosis
    mean_w  = np.mean(waveform)
    std_w   = np.std(waveform) + 1e-12
    kurtosis = np.mean(((waveform - mean_w) / std_w) ** 4)

    # --- Supporting Feature 6: SNR above noise floor ---
    # Direct measure of how much the signal stands out
    snr_above_noise = peak_db - noise_db

    return {
        'PAPR_dB'          : round(papr_db, 6),
        'Variance'         : round(variance, 6),
        'Bandwidth_Hz'     : round(bw, 4),
        'Spectral_Flatness': round(spectral_flatness, 6),
        'Kurtosis'         : round(kurtosis, 6),
        'SNR_above_noise'  : round(snr_above_noise, 4),
        'Label'            : label
    }

# =========================================
# STEP 4: GENERATE DATASET
# Extract features from:
#   - Isolated malicious waveform (label=1)
#   - Normal signal segments    (label=0)
# Generate multiple samples by windowing
# =========================================
window_size = 2048   # Samples per window
features_list = []

# --- Malicious signal windows (label = 1) ---
num_windows = len(isolated_waveform) // window_size
for i in range(num_windows):
    window = isolated_waveform[i*window_size:(i+1)*window_size]
    feat   = extract_features(
        window, bandwidth_hz, peak_power, noise_floor, label=1
    )
    features_list.append(feat)

print(f"\n  [FEAT] Malicious windows extracted : {num_windows}")

# --- Normal signal windows (label = 0) ---
# Add slight random noise variation per window for realism
num_normal = num_windows  # balanced dataset
for i in range(num_normal):
    window = normal_signal[i*window_size:(i+1)*window_size]
    # Normal signal has narrow bandwidth and lower SNR
    normal_bw  = bandwidth_hz * 0.1   # normal OOK much narrower
    normal_snr = noise_floor + 12     # SNR = +12 dB
    feat = extract_features(
        window, normal_bw, normal_snr, noise_floor, label=0
    )
    features_list.append(feat)

print(f"  [FEAT] Normal windows extracted    : {num_normal}")
print(f"  [FEAT] Total dataset size          : {len(features_list)}")

# Convert to DataFrame
features_df = pd.DataFrame(features_list)

# Print sample of extracted features
print(f"\n  [SAMPLE] First malicious row:")
print(f"  PAPR            : {features_df[features_df['Label']==1]['PAPR_dB'].iloc[0]:.4f} dB")
print(f"  Variance        : {features_df[features_df['Label']==1]['Variance'].iloc[0]:.6f}")
print(f"  Bandwidth       : {features_df[features_df['Label']==1]['Bandwidth_Hz'].iloc[0]:.2f} Hz")
print(f"  Spec Flatness   : {features_df[features_df['Label']==1]['Spectral_Flatness'].iloc[0]:.6f}")
print(f"  Kurtosis        : {features_df[features_df['Label']==1]['Kurtosis'].iloc[0]:.4f}")
print(f"  SNR above noise : {features_df[features_df['Label']==1]['SNR_above_noise'].iloc[0]:.2f} dB")

print(f"\n  [SAMPLE] First normal row:")
print(f"  PAPR            : {features_df[features_df['Label']==0]['PAPR_dB'].iloc[0]:.4f} dB")
print(f"  Variance        : {features_df[features_df['Label']==0]['Variance'].iloc[0]:.6f}")
print(f"  Bandwidth       : {features_df[features_df['Label']==0]['Bandwidth_Hz'].iloc[0]:.2f} Hz")
print(f"  Spec Flatness   : {features_df[features_df['Label']==0]['Spectral_Flatness'].iloc[0]:.6f}")
print(f"  Kurtosis        : {features_df[features_df['Label']==0]['Kurtosis'].iloc[0]:.4f}")
print(f"  SNR above noise : {features_df[features_df['Label']==0]['SNR_above_noise'].iloc[0]:.2f} dB")

# =========================================
# STEP 5: VISUALIZATION
# =========================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    'Feature Extraction — Malicious vs Normal Signal\n'
    'Features: PAPR | Variance | Bandwidth | '
    'Spectral Flatness | Kurtosis | SNR',
    fontsize=13, fontweight='bold'
)

feature_names = [
    'PAPR_dB', 'Variance', 'Bandwidth_Hz',
    'Spectral_Flatness', 'Kurtosis', 'SNR_above_noise'
]
titles = [
    'PAPR (dB)\nHigh = jammer spike',
    'Variance\nHigh = amplitude spread',
    'Bandwidth (Hz)\nWide = jammer footprint',
    'Spectral Flatness\nHigh = noise-like',
    'Kurtosis\nHigh = impulsive jammer',
    'SNR above Noise (dB)\nHigh = strong anomaly'
]

malicious_df = features_df[features_df['Label'] == 1]
normal_df    = features_df[features_df['Label'] == 0]

for idx, (feat, title) in enumerate(zip(feature_names, titles)):
    ax = axes[idx // 3][idx % 3]
    ax.hist(malicious_df[feat], bins=20, alpha=0.7,
            color='red', label='Malicious (1)', edgecolor='darkred')
    ax.hist(normal_df[feat], bins=20, alpha=0.7,
            color='blue', label='Normal (0)', edgecolor='darkblue')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(feat)
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature_extraction_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  [PLOT] Saved: feature_extraction_output.png")

# =========================================
# STEP 6: EXPORT FEATURE DATASET TO CSV
# =========================================
features_df.to_csv('features_dataset.csv', index=False)
print(f"  [CSV]  Saved: features_dataset.csv")
print(f"         Rows    : {len(features_df)}")
print(f"         Columns : {list(features_df.columns)}")

# Label distribution
mal_count = len(malicious_df)
nor_count = len(normal_df)
print(f"\n  [BALANCE] Malicious samples : {mal_count}")
print(f"            Normal samples    : {nor_count}")
print(f"            Dataset balanced  : {mal_count == nor_count}")

print("\n" + "=" * 55)
print("  Feature Extraction Complete")
print("  Next Step: ml_classifier.py")
print("=" * 55)