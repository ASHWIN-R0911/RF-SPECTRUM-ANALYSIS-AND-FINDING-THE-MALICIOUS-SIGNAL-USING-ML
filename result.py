# =========================================
# Final Output Dashboard (v2) - RF Signal Classification
# Project: RF Signal Classification and Malicious Signal Detection
#
# RF_MODE=synthetic (default)  single-source benchmark, group-aware split
#     bundle  : ensemble_bundle.pkl
#     results : multiclass_results.csv
#     dataset : rf_synthetic8_dataset.csv
# RF_MODE=hybrid               RadioML + Kaggle RSSI + synthetic, random split
#     bundle  : ensemble_bundle_hybrid.pkl
#     results : multiclass_hybrid_results.csv
#     dataset : rf_final_hybrid_dataset.csv
#
# Needs in the data folder: mixed_rf_signal.csv, cfar_output.csv (cacfar.py v2),
# the files of the chosen mode, and generate_multiclass_dataset_v2.py
# (its extract_features() is the SAME code used to build the training data).
#
# What changed vs v1
#  1. No hard-coded path (RF_DATA_DIR or script folder).
#  2. Models come from the saved ensemble bundle (scaler inside the pipelines,
#     class order checked, reject threshold from training). No legacy .pkl files.
#  3. Features use generate_multiclass_dataset_v2.extract_features (single source
#     of truth). v1 kept a copy and demodulated at the CFAR centre frequency, but
#     the model was trained with a fixed 204.8 kHz carrier.
#  4. Spectrum / bins follow cacfar.py v2 (N_FFT, Hann, power). v1 used the old
#     N//8 layout, so Bin_Low/Bin_High from the new CSV would map to wrong
#     frequencies.
#  5. Pipeline logic matches the paper: CFAR "no detection" -> benign, classifier
#     not invoked. All non-overlapping windows of the capture are classified and
#     their probabilities averaged. Low probability -> UNCERTAIN.
#  6. Removed claims the data cannot support: "433.92 MHz real band", "real:
#     RadioML/Kaggle" tags in synthetic mode, "confidence" (now model-estimated
#     probability). Panel 3 uses a log axis (features differ by orders of magnitude).
# =========================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import joblib
from scipy.signal import resample
import warnings
warnings.filterwarnings('ignore')

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
DATA_DIR = os.environ.get("RF_DATA_DIR", _HERE)
os.chdir(DATA_DIR)
sys.path.insert(0, DATA_DIR)
sys.path.insert(0, _HERE)

MODE = os.environ.get("RF_MODE", "synthetic").lower()
if MODE not in ("synthetic", "hybrid"):
    raise SystemExit("RF_MODE must be 'synthetic' or 'hybrid'")
FILES = {
    "synthetic": dict(bundle="ensemble_bundle.pkl", results="multiclass_results.csv",
                      dataset="rf_synthetic8_dataset.csv", out="final_output_dashboard.png"),
    "hybrid": dict(bundle="ensemble_bundle_hybrid.pkl", results="multiclass_hybrid_results.csv",
                   dataset="rf_final_hybrid_dataset.csv", out="final_output_dashboard_hybrid.png"),
}[MODE]
SHOW_PLOT = os.environ.get("RF_SHOW_PLOT", "1") == "1"

import generate_multiclass_dataset_v2 as G      # extract_features, FS, N_SAMPLES

print("=" * 60)
print(f"  Final Output Dashboard (v2) - mode: {MODE}")
print("=" * 60)

# =========================================
# LOAD
# =========================================
signal_df = pd.read_csv('mixed_rf_signal.csv')
cfar_df = pd.read_csv('cfar_output.csv')
feat_df = pd.read_csv(FILES["dataset"])
result_df = pd.read_csv(FILES["results"])
bundle = joblib.load(FILES["bundle"])

if 'N_FFT' not in cfar_df.columns:
    raise SystemExit("cfar_output.csv is from the old cacfar.py. Re-run cacfar.py v2 first.")

mixed_signal = signal_df['Mixed_Signal'].values
normal_signal = signal_df['Normal_Signal'].values
t = signal_df['Time_sec'].values
fs = 2.048e6

cfar_hit = str(cfar_df['Detection'].values[0]) == 'MALICIOUS'
n_fft = int(cfar_df['N_FFT'].values[0])
k_avg = int(cfar_df['Frames_Averaged'].values[0])
if cfar_hit:
    center_freq = float(cfar_df['Center_Freq_kHz'].values[0])
    bandwidth_hz = float(cfar_df['Bandwidth_Hz'].values[0])
    peak_power = float(cfar_df['Peak_Power_dB'].values[0])
    noise_floor = float(cfar_df['Noise_Floor_dB'].values[0])
    k_low = int(cfar_df['Bin_Low'].values[0])
    k_high = int(cfar_df['Bin_High'].values[0])

row = lambda m, c: result_df.loc[result_df['Model'] == m, c].values[0]
ens_acc, ens_prec = row('Ensemble', 'Accuracy'), row('Ensemble', 'Precision_macro')
ens_rec, ens_f1 = row('Ensemble', 'Recall_macro'), row('Ensemble', 'F1_macro')
svm_acc, rf_acc = row('SVM', 'Accuracy'), row('Random Forest', 'Accuracy')
split_txt = str(result_df['Split'].values[0]) if 'Split' in result_df.columns else 'n/a'

FEATURE_COLS = bundle['feature_cols']
LABEL_NAMES = {int(k): v for k, v in bundle['label_names'].items()}
MALICIOUS_LABELS = set(bundle['malicious_labels'])
REJECT_TH = float(bundle['reject_threshold'])
svm_pipe, rf_pipe = bundle['svm_pipeline'], bundle['rf_pipeline']
assert list(svm_pipe.classes_) == list(range(8)) and list(rf_pipe.classes_) == list(range(8)), \
    "model classes are not 0..7"
LEGIT_CLASSES = ['Normal_BPSK', 'Normal_QPSK', 'Normal_FM']
MALICIOUS_CLASSES = ['Jam_Tone', 'Jam_Sweep', 'Jam_Pulsed', 'Spoofed_Replay', 'Jam_Gaussian']
print(f"  [LOAD] bundle={FILES['bundle']}  split={split_txt}  CFAR detection={cfar_hit}")

# =========================================
# SPECTRUM (same method as cacfar.py v2)
# =========================================
def averaged_power_spectrum(x, n_fft, k):
    x = x - np.mean(x)
    n_frames = len(x) // n_fft
    k = max(1, min(k, n_frames))
    starts = np.unique(np.linspace(0, n_frames - 1, k).astype(int)) * n_fft
    win = np.hanning(n_fft)
    wp = np.sum(win ** 2)
    P = np.zeros(n_fft // 2 + 1)
    for s in starts:
        P += np.abs(np.fft.rfft(x[s:s + n_fft] * win)) ** 2 / wp
    return P / len(starts)

P = averaged_power_spectrum(mixed_signal, n_fft, k_avg)
fft_db = 10 * np.log10(P + 1e-18)
frequencies = np.fft.rfftfreq(n_fft, 1 / fs)

# =========================================
# CLASSIFY THE CAPTURE (only if CFAR flagged a region)
# =========================================
win_dur = G.N_SAMPLES / G.FS                          # training window: 1.024 ms
win_len = int(win_dur * fs)                           # samples at the capture rate
n_windows = max(1, len(mixed_signal) // win_len)
proba = None
if cfar_hit:
    feats = []
    for w_i in range(n_windows):
        seg = mixed_signal[w_i * win_len:(w_i + 1) * win_len]
        seg = resample(seg, G.N_SAMPLES)              # same duration -> same sample count
        feats.append(G.extract_features(seg))
    X = pd.DataFrame(feats, columns=FEATURE_COLS)
    proba = ((svm_pipe.predict_proba(X.values) + rf_pipe.predict_proba(X.values)) / 2).mean(axis=0)
    best = int(np.argmax(proba))
    best_p = float(proba[best])
    uncertain = best_p < REJECT_TH
    predicted_name = LABEL_NAMES[best]
    is_malicious = (best in MALICIOUS_LABELS) and not uncertain
    if uncertain:
        verdict = 'UNCERTAIN'
    else:
        verdict = 'MALICIOUS' if is_malicious else 'LEGITIMATE'
    print(f"  [PREDICT] {n_windows} window(s) -> {predicted_name}, model-estimated probability "
          f"{best_p*100:.2f}%  -> {verdict}")
else:
    best_p, uncertain, is_malicious = 0.0, False, False
    predicted_name, verdict = 'n/a', 'BENIGN (no CFAR detection)'
    print("  [PREDICT] CFAR found no region: capture declared benign, classifier not invoked")

# =========================================
# FIGURE
# =========================================
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#0d0d0d')
fig.text(0.5, 0.98, 'RF SIGNAL CLASSIFICATION & MALICIOUS SIGNAL DETECTION - FINAL OUTPUT',
         ha='center', va='top', fontsize=14, fontweight='bold', color='white')
fig.text(0.5, 0.955,
         '8 classes: BPSK | QPSK | FM | Tone | Sweep | Pulsed | Spoofed-Replay | Gaussian  -  '
         'CA-CFAR + SVM/Random Forest ensemble   [' + MODE + ' model]',
         ha='center', va='top', fontsize=9, color='#aaaaaa')
gs = gridspec.GridSpec(3, 3, figure=fig, top=0.875, bottom=0.05, left=0.05, right=0.97,
                       hspace=0.45, wspace=0.32, height_ratios=[1, 1, 0.55])


def style_ax(ax):
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    for s in ax.spines.values():
        s.set_edgecolor('#444444')
    ax.grid(True, alpha=0.2, color='#444444')


# Panel 1
ax1 = fig.add_subplot(gs[0, 0]); style_ax(ax1)
p = min(5000, len(mixed_signal))
ax1.plot(t[:p] * 1e3, mixed_signal[:p], color='#cc44ff', lw=0.5, label='Mixed signal')
ax1.plot(t[:p] * 1e3, normal_signal[:p], color='#4488ff', lw=0.5, alpha=0.5, label='Normal component')
ax1.set_title('Panel 1: Example mixed capture (synthetic)', color='white', fontsize=9)
ax1.set_xlabel('Time (ms)', color='#aaaaaa', fontsize=8); ax1.set_ylabel('Amplitude', color='#aaaaaa', fontsize=8)
ax1.legend(fontsize=7, facecolor='#2a2a2a', labelcolor='white')

# Panel 2
ax2 = fig.add_subplot(gs[0, 1]); style_ax(ax2)
ax2.plot(frequencies / 1e3, fft_db, color='white', lw=0.6, label=f'Power spectrum ({k_avg}-frame avg)')
if cfar_hit:
    ax2.axvspan(frequencies[k_low] / 1e3, frequencies[k_high] / 1e3, alpha=0.3, color='#ff4444',
                label=f'Region: BW {bandwidth_hz:.0f} Hz')
    ax2.axvline(center_freq, color='#ff4444', lw=2, label=f'Centre {center_freq:.1f} kHz')
    ax2.axhline(noise_floor, color='#ffaa00', lw=1, ls='--', label=f'Noise floor {noise_floor:.1f} dB')
    ax2.set_title(f'Panel 2: FFT + CA-CFAR detection\n(bin width {fs/n_fft/1e3:.1f} kHz)', color='white', fontsize=9)
else:
    ax2.set_title('Panel 2: FFT + CA-CFAR - no region flagged', color='white', fontsize=9)
ax2.set_xlabel('Frequency (kHz)', color='#aaaaaa', fontsize=8); ax2.set_ylabel('Power (dB)', color='#aaaaaa', fontsize=8)
ax2.legend(fontsize=7, facecolor='#2a2a2a', labelcolor='white')

# Panel 3 (log axis)
ax3 = fig.add_subplot(gs[1, 0]); style_ax(ax3)
top_features = ['Variance', 'Order2_PAR', 'Spectral_Entropy', 'Echo_Autocorr_Strength']
legit_f = feat_df[feat_df['Label_Name'].isin(LEGIT_CLASSES)][top_features].mean()
mal_f = feat_df[feat_df['Label_Name'].isin(MALICIOUS_CLASSES)][top_features].mean()
x = np.arange(len(top_features)); w = 0.35
ax3.bar(x - w / 2, legit_f.values, w, label='Legitimate (avg)', color='#4488ff', edgecolor='#224488')
ax3.bar(x + w / 2, mal_f.values, w, label='Malicious (avg)', color='#ff4444', edgecolor='#882222')
ax3.set_yscale('log')
ax3.set_xticks(x)
ax3.set_xticklabels(['Variance', 'Order2\nPAR', 'Spectral\nEntropy', 'Echo\nAutocorr'], color='#aaaaaa', fontsize=8)
ttl = 'Panel 3: Feature means, legitimate vs malicious (log axis)'
if MODE == 'hybrid':
    ttl += '\nhybrid data: classes come from different sources'
ax3.set_title(ttl, color='white', fontsize=9)
ax3.set_ylabel('Mean feature value', color='#aaaaaa', fontsize=8)
ax3.legend(fontsize=7, facecolor='#2a2a2a', labelcolor='white')

# Panel 4
ax4 = fig.add_subplot(gs[1, 1]); style_ax(ax4)
accs = [svm_acc * 100, rf_acc * 100, ens_acc * 100]
bars = ax4.bar(['SVM', 'Random\nForest', 'Ensemble\n(avg)'], accs, color=['#4488ff', '#44bb44', '#ff4444'],
               edgecolor='#333333', width=0.5)
ax4.set_ylim([0, 108])
ax4.set_title(f'Panel 4: Test accuracy (8 classes)\nsplit: {split_txt}', color='white', fontsize=9)
ax4.set_ylabel('Accuracy (%)', color='#aaaaaa', fontsize=8)
for bar, val in zip(bars, accs):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{val:.2f}%', ha='center',
             fontweight='bold', fontsize=10, color='white')

# Panel 5 (summary card)
ax5 = fig.add_subplot(gs[0:2, 2]); ax5.set_facecolor('#111111'); ax5.axis('off')
vcol = {'MALICIOUS': '#ff4444', 'LEGITIMATE': '#44ff88', 'UNCERTAIN': '#ffcc00'}.get(verdict, '#44ff88')
header = {'MALICIOUS': 'MALICIOUS SIGNAL PRESENT', 'LEGITIMATE': 'NO MALICIOUS SIGNAL',
          'UNCERTAIN': 'UNCERTAIN - LOW PROBABILITY'}.get(verdict, 'BENIGN: NO CFAR DETECTION')
if cfar_hit:
    detail = f'Most likely class: {predicted_name}  (top 3 below)'
    prob_line = f'Model-estimated probability: {best_p*100:.1f}%'
else:
    detail, prob_line = 'Classifier not invoked', ''
src = (lambda s_syn, s_real: s_syn) if MODE == 'synthetic' else (lambda s_syn, s_real: s_real)
lines = [
    ('DETECTION RESULT', '#ff4444', 13, 'bold'), ('', 'white', 4, 'normal'),
    (header, vcol, 10, 'bold'), (detail, vcol, 9, 'normal'), (prob_line, vcol, 9, 'normal'),
    *[(f'   {LABEL_NAMES[i]:<15}{proba[i]*100:5.1f}%', vcol, 8, 'normal')
      for i in (np.argsort(proba)[::-1][:3] if cfar_hit else [])],
    ('', 'white', 4, 'normal'),
    ('CLASSES THE MODEL KNOWS', '#ffffff', 10, 'bold'), ('-' * 28, '#555555', 8, 'normal'),
    ('  Legitimate:', '#4488ff', 9, 'bold'),
    (src('    - BPSK  (synthetic)', '    - BPSK  (RadioML)'), '#ffffff', 8, 'normal'),
    (src('    - QPSK  (synthetic)', '    - QPSK  (RadioML)'), '#ffffff', 8, 'normal'),
    (src('    - FM  (synthetic)', '    - FM  (RadioML)'), '#ffffff', 8, 'normal'),
    ('', 'white', 3, 'normal'), ('  Malicious:', '#ff4444', 9, 'bold'),
    (src('    - Tone jammer  (synthetic)', '    - Tone jammer  (Kaggle RSSI)'), '#ffffff', 8, 'normal'),
    (src('    - Gaussian jammer  (synthetic)', '    - Gaussian jammer  (Kaggle RSSI)'), '#ffffff', 8, 'normal'),
    ('    - Sweep jammer  (synthetic)', '#ffffff', 8, 'normal'),
    ('    - Pulsed jammer  (synthetic)', '#ffffff', 8, 'normal'),
    ('    - Spoofed replay  (synthetic)', '#ffffff', 8, 'normal'),
    ('', 'white', 4, 'normal'), ('-' * 28, '#555555', 8, 'normal'),
    ('EXAMPLE CAPTURE (Panels 1-2)', '#aaaaaa', 9, 'bold'), ('-' * 28, '#555555', 8, 'normal'),
]
if cfar_hit:
    lines += [(f'Centre freq : {center_freq:.2f} kHz (baseband)', '#ffffff', 9, 'normal'),
              (f'Bandwidth   : {bandwidth_hz:.0f} Hz', '#ffffff', 9, 'normal'),
              (f'Peak power  : {peak_power:.2f} dB', '#ffffff', 9, 'normal')]
else:
    lines += [('No region above CA-CFAR threshold', '#ffffff', 9, 'normal')]
lines += [('', 'white', 4, 'normal'), ('-' * 28, '#555555', 8, 'normal'),
          ('ENSEMBLE SCORES (test set, macro-avg)', '#44ff88', 9, 'bold'), ('-' * 28, '#555555', 8, 'normal'),
          (f'Accuracy  : {ens_acc*100:.2f}%', '#44ff88', 9, 'normal'),
          (f'Precision : {ens_prec*100:.2f}%', '#44ff88', 9, 'normal'),
          (f'Recall    : {ens_rec*100:.2f}%', '#44ff88', 9, 'normal'),
          (f'F1 score  : {ens_f1*100:.2f}%', '#44ff88', 9, 'normal'),
          ('', 'white', 3, 'normal'),
          ('Valid for SNR ~4-10 dB, JSR 5-20 dB (synthetic)' if MODE == 'synthetic'
           else 'Hybrid-trained: source-specific, not validated', '#ffcc00', 7, 'normal'),
          ('outside this range accuracy drops (see paper)' if MODE == 'synthetic'
           else 'on new captures (see paper, Sec. V-6)', '#ffcc00', 7, 'normal')]
y = 0.97
for text, color, size, weight in lines:
    ax5.text(0.06, y, text, transform=ax5.transAxes, color=color, fontsize=size, fontweight=weight,
             va='top', fontfamily='monospace')
    y -= 0.027

# Panel 6 (pipeline)
ax6 = fig.add_subplot(gs[2, :]); ax6.set_facecolor('#111111'); ax6.axis('off')
ax6.set_title('Detection pipeline - signal flow', color='white', fontsize=10, fontweight='bold', pad=6)
final_label = (f'{verdict}\n{predicted_name}\n{best_p*100:.1f}%' if cfar_hit else 'BENIGN\n(no CFAR region)')
final_color = {'MALICIOUS': '#cc2222', 'LEGITIMATE': '#228822', 'UNCERTAIN': '#aa8800'}.get(verdict, '#228822')
steps = [('Captured signal\n(synthetic,\nbaseband)', '#9933cc'), ('FFT +\nframe averaging', '#2255cc'),
         ('CA-CFAR\nregion detection', '#cc6600'), ('13 features\nPAPR | Var |\nOrder2/4 PAR | ...', '#0088aa'),
         ('SVM + Random Forest\nprobability average', '#228822'), (final_label, final_color)]
bw, bh, by, xs = 0.13, 0.62, 0.15, 0.025
for i, (label, color) in enumerate(steps):
    xx = xs + i * (bw + 0.038)
    ax6.add_patch(FancyBboxPatch((xx, by), bw, bh, boxstyle='round,pad=0.02', facecolor=color,
                                 edgecolor='white', alpha=0.9, transform=ax6.transAxes, clip_on=False))
    ax6.text(xx + bw / 2, by + bh / 2, label, transform=ax6.transAxes, ha='center', va='center',
             fontsize=9, fontweight='bold', color='white', multialignment='center')
    if i < len(steps) - 1:
        ax_x = xx + bw + 0.004
        ax6.annotate('', xy=(ax_x + 0.026, by + bh / 2), xytext=(ax_x, by + bh / 2),
                     xycoords='axes fraction', textcoords='axes fraction',
                     arrowprops=dict(arrowstyle='->', color='white', lw=2))

plt.savefig(FILES["out"], dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
if SHOW_PLOT:
    plt.show()
print(f"\n  [DONE] {FILES['out']} saved")
print("=" * 60)
print(f"  Capture verdict : {verdict}" + (f" ({predicted_name}, {best_p*100:.2f}% model-estimated probability)" if cfar_hit else ""))
print(f"  Test accuracy   : {ens_acc*100:.2f}% | Precision {ens_prec*100:.2f}% | Recall {ens_rec*100:.2f}%  (split: {split_txt})")
print("=" * 60)