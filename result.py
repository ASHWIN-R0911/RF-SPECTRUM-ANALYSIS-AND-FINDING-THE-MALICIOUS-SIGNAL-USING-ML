# =========================================
# Final Output Dashboard - Multi-Class Update
# Project: RF Signal Classification and Malicious Signal Detection
# =========================================

import os
os.chdir(r'E:\D\projectsml')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  Final Output Dashboard — RF Signal Classification")
print("  + Malicious Signal Detection")
print("=" * 60)

# =========================================
# LOAD ALL RESULTS
# =========================================
signal_df  = pd.read_csv('mixed_rf_signal.csv')
cfar_df    = pd.read_csv('cfar_output.csv')
feat_df    = pd.read_csv('rf_final_hybrid_dataset.csv')    # UPDATED: real+synthetic hybrid dataset
result_df  = pd.read_csv('multiclass_results.csv')          # NEW: multi-class results

mixed_signal     = signal_df['Mixed_Signal'].values
normal_signal    = signal_df['Normal_Signal'].values
t                = signal_df['Time_sec'].values
fs               = 2.048e6

center_freq  = cfar_df['Center_Freq_kHz'].values[0]
bandwidth_hz = cfar_df['Bandwidth_Hz'].values[0]
peak_power   = cfar_df['Peak_Power_dB'].values[0]
noise_floor  = cfar_df['Noise_Floor_dB'].values[0]
k_low        = int(cfar_df['Bin_Low'].values[0])
k_high       = int(cfar_df['Bin_High'].values[0])

# Macro-averaged multi-class metrics (8 classes, not binary)
ens_acc  = result_df[result_df['Model']=='Ensemble']['Accuracy'].values[0]
ens_prec = result_df[result_df['Model']=='Ensemble']['Precision_macro'].values[0]
ens_rec  = result_df[result_df['Model']=='Ensemble']['Recall_macro'].values[0]
ens_f1   = result_df[result_df['Model']=='Ensemble']['F1_macro'].values[0]
svm_acc  = result_df[result_df['Model']=='SVM']['Accuracy'].values[0]
rf_acc   = result_df[result_df['Model']=='Random Forest']['Accuracy'].values[0]

# Class groups
LEGIT_CLASSES = ['Normal_BPSK', 'Normal_QPSK', 'Normal_FM']
MALICIOUS_CLASSES = ['Jam_Tone', 'Jam_Sweep', 'Jam_Pulsed', 'Spoofed_Replay', 'Jam_Gaussian']

print("  [LOAD] All files loaded (8-class multi-signal dataset)")

# =========================================
# RECOMPUTE FFT + ISOLATE WAVEFORM
# (unchanged - this is your live captured jamming example)
# =========================================
num_avg = 8
seg_len = len(mixed_signal) // num_avg
fft_avg = np.zeros(seg_len // 2)
for i in range(num_avg):
    seg      = mixed_signal[i*seg_len:(i+1)*seg_len]
    fft_avg += np.abs(np.fft.fft(seg)[:seg_len//2])
fft_avg    /= num_avg
fft_db      = 20 * np.log10(fft_avg + 1e-12)
frequencies = np.fft.fftfreq(seg_len, 1/fs)[:seg_len//2]
freq_res    = fs / seg_len

N            = len(mixed_signal)
scale        = N / seg_len
k_low_full   = int(k_low  * scale)
k_high_full  = int(k_high * scale)
fft_full     = np.fft.fft(mixed_signal)
fft_isolated = np.zeros(N, dtype=complex)
fft_isolated[k_low_full:k_high_full+1] = fft_full[k_low_full:k_high_full+1]
fft_isolated[N-k_high_full:N-k_low_full+1] = fft_full[N-k_high_full:N-k_low_full+1]
isolated_waveform = np.real(np.fft.ifft(fft_isolated))

# =========================================
# CLASSIFY THE ACTUAL CAPTURED SIGNAL
# Loads the trained models and runs them on THIS specific signal
# (mixed_signal from mixed_rf_signal.csv), rather than just showing
# aggregate test-set stats. This is what makes the verdict dynamic:
# if the captured signal is legitimate (BPSK/QPSK/FM), it says so
# instead of always claiming malicious detection.
# =========================================
import joblib
from scipy.stats import kurtosis, skew
from scipy.signal import welch, find_peaks

svm_model = joblib.load('svm_model.pkl')
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# UPDATED: Phase_State_Count replaced with Order2_PAR/Order4_PAR - the old
# phase-histogram peak-counting feature was noise-fragile and gave heavily
# overlapping BPSK/QPSK distributions (~90% recall each). Order2_PAR/
# Order4_PAR use the M-th power nonlinearity technique from automatic
# modulation classification instead - verified on real RadioML data at
# ~98-99% recall. See load_radioml_dataset.py / generate_multiclass_dataset.py
# for the full explanation.
FEATURE_COLS = ['PAPR_dB', 'Variance', 'Spectral_Flatness', 'Kurtosis',
                'Spectral_Entropy', 'Skewness', 'ZCR',
                'Spectral_Peak_Count', 'Envelope_Duty_Cycle',
                'Echo_Autocorr_Strength', 'Order2_PAR', 'Order4_PAR', 'Phase_Variance']
LABEL_NAMES = {0: 'BPSK', 1: 'QPSK', 2: 'FM',
               3: 'Tone-Jam', 4: 'Sweep-Jam', 5: 'Pulsed-Jam',
               6: 'Spoofed-Replay', 7: 'Gaussian-Jam'}
MALICIOUS_LABELS = {3, 4, 5, 6, 7}
CARRIER_FREQ_FOR_DEMOD = center_freq * 1e3  # from cfar_output.csv, in Hz


def extract_features_from_signal(signal, fs, carrier_freq):
    """Same 13-feature extraction used in training - must match exactly."""
    peak_power = np.max(np.abs(signal)) ** 2
    avg_power = np.mean(signal ** 2)
    papr_db = 10 * np.log10(peak_power / (avg_power + 1e-12))
    variance = np.var(signal)

    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    psd = psd + 1e-12
    gm = np.exp(np.mean(np.log(psd)))
    am = np.mean(psd)
    spectral_flatness = gm / am
    kurt = kurtosis(signal)

    psd_norm = psd / np.sum(psd)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    spectral_entropy /= np.log2(len(psd_norm))

    skewness = skew(signal)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(signal))) > 0)
    zcr = zero_crossings / (len(signal) - 1)

    threshold = np.max(psd) * 0.3
    peaks, _ = find_peaks(psd, height=threshold)
    spectral_peak_count = len(peaks)

    envelope = np.abs(signal)
    window = 8
    smooth_env = np.convolve(envelope, np.ones(window) / window, mode="same")
    env_threshold = np.max(smooth_env) * 0.5
    duty_cycle = np.mean(smooth_env >= env_threshold)

    sig_zero_mean = signal - np.mean(signal)
    zero_lag_energy = np.sum(sig_zero_mean ** 2) + 1e-12
    max_lag = min(200, len(signal) // 2)
    autocorr_vals = []
    for lag in range(20, max_lag):
        ac = np.sum(sig_zero_mean[:-lag] * sig_zero_mean[lag:])
        autocorr_vals.append(abs(ac) / zero_lag_energy)
    echo_autocorr_strength = max(autocorr_vals) if autocorr_vals else 0.0

    # --- Order2_PAR / Order4_PAR (BPSK vs QPSK discriminator) ---
    # Coherent I/Q demodulation at the known carrier frequency gives a
    # complex baseband signal I_f + j*Q_f. Raising it to the 2nd/4th power
    # strips off M-PSK phase modulation once the power matches the
    # constellation order (BPSK collapses to a sharp spectral tone at
    # BOTH order-2 and order-4; QPSK only collapses at order-4). We
    # measure "sharp tone vs spread spectrum" via peak-to-average power
    # ratio (PAR) of the FFT - same technique used in training.
    n = len(signal)
    t_local = np.arange(n) / fs
    I = signal * np.cos(2 * np.pi * carrier_freq * t_local)
    Q = -signal * np.sin(2 * np.pi * carrier_freq * t_local)
    w = 32
    kernel = np.ones(w) / w
    I_f = np.convolve(I, kernel, mode="same")
    Q_f = np.convolve(Q, kernel, mode="same")

    iq_baseband = I_f + 1j * Q_f

    def _mth_power_par(z, m):
        zm = z ** m
        Zf = np.abs(np.fft.fft(zm)) ** 2
        return np.max(Zf) / (np.mean(Zf) + 1e-12)

    order2_par = _mth_power_par(iq_baseband, 2)
    order4_par = _mth_power_par(iq_baseband, 4)

    phase = np.arctan2(Q_f, I_f)
    phase_mod = np.mod(phase, 2 * np.pi)
    phase_variance = np.var(phase_mod)

    return [papr_db, variance, spectral_flatness, kurt, spectral_entropy,
            skewness, zcr, spectral_peak_count, duty_cycle,
            echo_autocorr_strength, order2_par, order4_par, phase_variance]


from scipy.signal import resample

TRAIN_FS = 1_000_000       # sample rate used in generate_multiclass_dataset.py
TRAIN_N_SAMPLES = 1024     # window length used in generate_multiclass_dataset.py

# Your captured signal (mixed_rf_signal.csv) may be a different sample rate
# and length than what the model was trained on. Feeding a mismatched
# window directly gives unreliable features (spectral peak count, phase
# clustering, and autocorrelation lag range are all scale-sensitive to
# both sample rate and window length). Fix: take a window covering the
# SAME real-world duration as training, then resample it to the SAME
# sample count, so features are computed on a like-for-like basis.
train_window_duration = TRAIN_N_SAMPLES / TRAIN_FS          # seconds
captured_window_samples = int(train_window_duration * fs)   # samples at fs
captured_window_samples = min(captured_window_samples, len(mixed_signal))
signal_window = mixed_signal[:captured_window_samples]
signal_window_resampled = resample(signal_window, TRAIN_N_SAMPLES)

captured_features = extract_features_from_signal(
    signal_window_resampled, TRAIN_FS, CARRIER_FREQ_FOR_DEMOD)
captured_features_scaled = scaler.transform([captured_features])

svm_p = svm_model.predict_proba(captured_features_scaled)[0]
rf_p = rf_model.predict_proba(captured_features_scaled)[0]
combined_proba = (svm_p + rf_p) / 2.0
predicted_label = int(np.argmax(combined_proba))
predicted_name = LABEL_NAMES[predicted_label]
predicted_confidence = combined_proba[predicted_label] * 100
is_malicious = predicted_label in MALICIOUS_LABELS

print(f"  [PREDICT] Captured signal classified as: {predicted_name} "
      f"({predicted_confidence:.2f}% confidence)")
print(f"  [PREDICT] Malicious: {'YES' if is_malicious else 'NO'}")

print("  [FFT]  Recomputed successfully")

# =========================================
# LAYOUT: 3 columns, 3 rows (unchanged structure)
# =========================================
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#0d0d0d')

fig.text(0.5, 0.98,
         'RF SIGNAL CLASSIFICATION & MALICIOUS SIGNAL DETECTION — FINAL OUTPUT',
         ha='center', va='top',
         fontsize=14, fontweight='bold', color='white')
fig.text(0.5, 0.955,
         '8-Class Detection: BPSK | QPSK | FM | Tone-Jam | Sweep-Jam | '
         'Pulsed-Jam | Spoofed-Replay | Gaussian-Jam  —  SVM + Random Forest Ensemble',
         ha='center', va='top', fontsize=9, color='#aaaaaa')

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    top=0.92, bottom=0.05,
    left=0.05, right=0.97,
    hspace=0.45, wspace=0.32,
    height_ratios=[1, 1, 0.55]
)

def style_ax(ax):
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    ax.grid(True, alpha=0.2, color='#444444')

# ─────────────────────────────────────────
# PANEL 1: Mixed Signal (unchanged - live captured example)
# ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1)
p = min(5000, len(mixed_signal))
ax1.plot(t[:p]*1e3, mixed_signal[:p],
         color='#cc44ff', linewidth=0.5, label='Mixed Signal')
ax1.plot(t[:p]*1e3, normal_signal[:p],
         color='#4488ff', linewidth=0.5, alpha=0.5, label='Normal')
ax1.set_title('Panel 1: Captured RF Signal Example\n(Normal + Malicious)',
              color='white', fontsize=9)
ax1.set_xlabel('Time (ms)', color='#aaaaaa', fontsize=8)
ax1.set_ylabel('Amplitude', color='#aaaaaa', fontsize=8)
ax1.legend(fontsize=7, facecolor='#2a2a2a', labelcolor='white')

# ─────────────────────────────────────────
# PANEL 2: FFT + CFAR (unchanged - live captured example)
# ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2)
ax2.plot(frequencies/1e3, fft_db,
         color='white', linewidth=0.6, label='FFT Spectrum')
ax2.axvline(x=center_freq, color='#ff4444',
            linewidth=2, label=f'Detected: {center_freq:.1f} kHz')
ax2.axvspan(frequencies[k_low]/1e3, frequencies[k_high]/1e3,
            alpha=0.3, color='#ff4444', label=f'BW: {bandwidth_hz:.0f} Hz')
ax2.axhline(y=noise_floor, color='#ffaa00', linewidth=1,
            linestyle='--', label=f'Noise: {noise_floor:.1f} dB')
ax2.set_title('Panel 2: FFT + CA-CFAR Detection\n(Example Capture)',
              color='white', fontsize=9)
ax2.set_xlabel('Frequency (kHz)', color='#aaaaaa', fontsize=8)
ax2.set_ylabel('Power (dB)', color='#aaaaaa', fontsize=8)
ax2.legend(fontsize=7, facecolor='#2a2a2a', labelcolor='white')

# ─────────────────────────────────────────
# PANEL 3: Feature Comparison — Legitimate vs Malicious classes
# (UPDATED: averaged across 3 legit + 5 malicious classes, and
#  Phase_State_Count swapped for Order2_PAR since that column no
#  longer exists in rf_final_hybrid_dataset.csv)
# ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3)

top_features = ['Variance', 'Order2_PAR', 'Spectral_Entropy', 'Echo_Autocorr_Strength']
legit_f = feat_df[feat_df['Label_Name'].isin(LEGIT_CLASSES)][top_features].mean()
mal_f   = feat_df[feat_df['Label_Name'].isin(MALICIOUS_CLASSES)][top_features].mean()

x  = np.arange(len(top_features))
w  = 0.35
ax3.bar(x-w/2, legit_f.values, w, label='Legitimate (avg)',
        color='#4488ff', edgecolor='#224488')
ax3.bar(x+w/2, mal_f.values, w, label='Malicious (avg)',
        color='#ff4444', edgecolor='#882222')
ax3.set_xticks(x)
ax3.set_xticklabels(['Variance', 'Order2\nPAR', 'Spectral\nEntropy', 'Echo\nAutocorr'],
                    color='#aaaaaa', fontsize=8)
ax3.set_title('Panel 3: Feature Comparison\nLegitimate vs Malicious (avg across classes)',
              color='white', fontsize=9)
ax3.set_ylabel('Feature Value', color='#aaaaaa', fontsize=8)
ax3.legend(fontsize=7, facecolor='#2a2a2a', labelcolor='white')

# ─────────────────────────────────────────
# PANEL 4: Model Accuracy Comparison (UPDATED: multi-class macro accuracy)
# ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4)
models = ['SVM', 'Random\nForest', 'Ensemble\n(Final)']
accs   = [svm_acc*100, rf_acc*100, ens_acc*100]
colors = ['#4488ff', '#44bb44', '#ff4444']
bars   = ax4.bar(models, accs, color=colors,
                 edgecolor='#333333', width=0.5)
ax4.set_ylim([0, 105])
ax4.set_title('Panel 4: Model Accuracy Comparison\n(8-Class Multi-Signal)',
              color='white', fontsize=9)
ax4.set_ylabel('Accuracy (%)', color='#aaaaaa', fontsize=8)
for bar, val in zip(bars, accs):
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             f'{val:.1f}%', ha='center',
             fontweight='bold', fontsize=11, color='white')

# ─────────────────────────────────────────
# PANEL 5: Summary Card — spans rows 0+1
# (UPDATED: full 8-class breakdown instead of single binary verdict)
# ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[0:2, 2])
ax5.set_facecolor('#111111')
ax5.axis('off')
for spine in ax5.spines.values():
    spine.set_edgecolor('#ff4444')

verdict_color = '#ff4444' if is_malicious else '#44ff88'
verdict_header = '⚠ MALICIOUS SIGNAL PRESENT' if is_malicious else '✓ NO MALICIOUS SIGNAL'
verdict_detail = f'Type: {predicted_name}' if is_malicious else f'Signal Type: {predicted_name} (legitimate)'

lines = [
    ('DETECTION RESULT',                '#ff4444',  13,  'bold'),
    ('',                                'white',     4,  'normal'),
    (verdict_header,                    verdict_color, 11, 'bold'),
    (verdict_detail,                    verdict_color, 9,  'normal'),
    (f'Confidence: {predicted_confidence:.2f}%', verdict_color, 9, 'normal'),
    ('',                                'white',     4,  'normal'),
    ('SIGNAL CLASSES THE MODEL KNOWS',  '#ffffff',  10,  'bold'),
    ('─' * 28,                          '#555555',   8,  'normal'),
    ('  Legitimate Signal Types:',      '#4488ff',   9,  'bold'),
    ('    • BPSK  (real: RadioML)', '#ffffff', 8, 'normal'),
    ('    • QPSK  (real: RadioML)', '#ffffff', 8, 'normal'),
    ('    • FM  (real: RadioML)',   '#ffffff', 8, 'normal'),
    ('',                                'white',     3,  'normal'),
    ('  Malicious Signal Types:',       '#ff4444',   9,  'bold'),
    ('    • Tone Jammer  (real: Kaggle)',      '#ffffff', 8, 'normal'),
    ('    • Gaussian-Noise Jammer  (real: Kaggle)', '#ffffff', 8, 'normal'),
    ('    • Sweep/Chirp Jammer  (synthetic)','#ffffff', 8, 'normal'),
    ('    • Pulsed Jammer  (synthetic)',    '#ffffff', 8, 'normal'),
    ('    • Spoofed/Replay  (synthetic)',   '#ffffff', 8, 'normal'),
    ('',                                'white',     4,  'normal'),
    ('─' * 28,                          '#555555',   8,  'normal'),
    ('EXAMPLE CAPTURE (Panels 1-2)',    '#aaaaaa',   9,  'bold'),
    ('─' * 28,                          '#555555',   8,  'normal'),
    (f'Center Freq  : {center_freq:.2f} kHz',
                                        '#ffffff',   9,  'normal'),
    ('Real Band    : 433.92 MHz ISM',   '#ffffff',   9,  'normal'),
    (f'Bandwidth    : {bandwidth_hz:.0f} Hz',
                                        '#ffffff',   9,  'normal'),
    (f'Peak Power   : {peak_power:.2f} dB',
                                        '#ffffff',   9,  'normal'),
    ('',                                'white',     4,  'normal'),
    ('─' * 28,                          '#555555',   8,  'normal'),
    ('ML ENSEMBLE SCORES (test set, macro-avg)', '#44ff88', 9, 'bold'),
    ('─' * 28,                          '#555555',   8,  'normal'),
    (f'Accuracy   :  {ens_acc*100:.1f}%',
                                        '#44ff88',   9,  'normal'),
    (f'Precision  :  {ens_prec*100:.1f}%',
                                        '#44ff88',   9,  'normal'),
    (f'Recall     :  {ens_rec*100:.1f}%',
                                        '#44ff88',   9,  'normal'),
    (f'F1 Score   :  {ens_f1*100:.1f}%',
                                        '#44ff88',   9,  'normal'),
]

y = 0.97
for text, color, size, weight in lines:
    ax5.text(0.06, y, text,
             transform=ax5.transAxes,
             color=color, fontsize=size,
             fontweight=weight,
             verticalalignment='top',
             fontfamily='monospace')
    y -= 0.028   # slightly tighter to fit the extra Gaussian-Jam line

# ─────────────────────────────────────────
# PANEL 6: Pipeline Flow — full bottom row
# (UPDATED: reflects multi-class output, real accuracy not hardcoded)
# ─────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, :])
ax6.set_facecolor('#111111')
ax6.axis('off')
ax6.set_title('Complete Detection Pipeline — Signal Flow',
              color='white', fontsize=10,
              fontweight='bold', pad=6)

final_box_label = (f'MALICIOUS\n{predicted_name}\n\u2713 {predicted_confidence:.1f}%' if is_malicious
                    else f'CLEAN SIGNAL\n{predicted_name}\n\u2713 {predicted_confidence:.1f}%')
final_box_color = '#cc2222' if is_malicious else '#228822'

steps = [
    ('RF Signal\nCapture\n433.92 MHz', '#9933cc'),
    ('FFT\nTransform\n(Averaged)',   '#2255cc'),
    ('CA-CFAR +\nFeature\nExtraction','#cc6600'),
    ('13 Features\nPAPR|Var|Order2/4\n|Echo|...','#0088aa'),
    ('SVM +\nRandom Forest\n(Ensemble)','#228822'),
    (final_box_label,  final_box_color),
]

bw, bh = 0.13, 0.62
by     = 0.15
xs     = 0.025

for i, (label, color) in enumerate(steps):
    x = xs + i * (bw + 0.038)
    ax6.add_patch(FancyBboxPatch(
        (x, by), bw, bh,
        boxstyle='round,pad=0.02',
        facecolor=color, edgecolor='white',
        alpha=0.9, transform=ax6.transAxes,
        clip_on=False
    ))
    ax6.text(x + bw/2, by + bh/2, label,
             transform=ax6.transAxes,
             ha='center', va='center',
             fontsize=9, fontweight='bold',
             color='white', multialignment='center')
    if i < len(steps)-1:
        ax_x = x + bw + 0.004
        ax6.annotate('',
            xy=(ax_x+0.026, by+bh/2),
            xytext=(ax_x, by+bh/2),
            xycoords='axes fraction',
            textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='white', lw=2)
        )

plt.savefig('final_output_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor='#0d0d0d')
plt.show()

print("\n  [DONE] final_output_dashboard.png saved")
print("=" * 60)
print("  PROJECT COMPLETE — ALL STEPS DONE")
print(f"  Captured Signal Verdict : {'MALICIOUS - ' + predicted_name if is_malicious else 'CLEAN (' + predicted_name + ')'} "
      f"({predicted_confidence:.2f}% confidence)")
print(f"  Model Test Accuracy     : {ens_acc*100:.1f}% | Precision: {ens_prec*100:.1f}% | Recall: {ens_rec*100:.1f}%")
print("=" * 60)