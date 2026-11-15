# =========================================
# Feature Extraction
# Project: Malicious RF Signal Detection
# Input  : mixed_rf_signal.csv
#           cfar_output.csv
# Output : features_dataset.csv
# =========================================

"""
Feature Extraction  (v2)
Project : Malicious RF Signal Detection
Input   : mixed_rf_signal.csv  (Time_sec, Mixed_Signal, Normal_Signal, ...)
          cfar_output.csv      (from cacfar.py v2)
Output  : features_dataset.csv

What changed vs v1
  1. Reads Detection from cfar_output.csv and stops if CFAR found nothing.
  2. Uses Bin_Low_Full / Bin_High_Full written by cacfar.py v2
     (v1 re-derived them with a hard-coded N // 8, which breaks when N_FFT changes).
  3. LABEL LEAKAGE REMOVED.  v1 gave every normal window
        bandwidth = CFAR_bandwidth * 0.1   and   SNR = noise_floor + 12
     and every malicious window the CFAR values, i.e. two columns were
     constants that simply encoded the label. Now Bandwidth_Hz and
     Peak_to_Noise_Level_dB are MEASURED from each window with the same function
     for both classes.
  4. Both classes go through the SAME processing. v1 band-passed the
     malicious class but used the raw normal signal, so a classifier could
     learn "was it filtered" instead of "is it malicious". Now the normal
     record is band-passed around its own strongest region with the same rule.
  5. Adds Block_ID (contiguous time block) so ml_classifier.py can split by
     block and avoid train/test windows that are neighbours in time.
  6. No hard-coded E:\\ path.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================
# CONFIG
# =========================================
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
DATA_DIR = os.environ.get("RF_DATA_DIR", _HERE)

FS            = 2.048e6
WINDOW_SIZE   = 2048        # samples per window
BW_DROP_DB    = 10.0        # bandwidth = bins within 10 dB of the window peak
DC_IGNORE     = 3           # ignore bins next to 0 Hz
BLOCKS        = 10          # number of contiguous time blocks (for group split)
NORMAL_NOISE_STD = None     # set e.g. 0.5 if Normal_Signal is noise-free but
                            # Mixed_Signal contains noise (keeps classes comparable)
SHOW_PLOT     = True


def path(name):
    return os.path.join(DATA_DIR, name)


print("=" * 55)
print("  Feature Extraction (v2)")
print("=" * 55)

# =========================================
# STEP 1: LOAD SIGNALS AND CFAR RESULT
# =========================================
df = pd.read_csv(path("mixed_rf_signal.csv"))
mixed_signal = df["Mixed_Signal"].values.astype(float)
normal_signal = df["Normal_Signal"].values.astype(float)

cfar_df = pd.read_csv(path("cfar_output.csv"))
if str(cfar_df["Detection"].values[0]) != "MALICIOUS":
    print("  [STOP] CFAR reported no detection - nothing to extract.")
    sys.exit(2)

center_freq = float(cfar_df["Center_Freq_kHz"].values[0])
cfar_bw = float(cfar_df["Bandwidth_Hz"].values[0])
k_low_full = int(cfar_df["Bin_Low_Full"].values[0])
k_high_full = int(cfar_df["Bin_High_Full"].values[0])
cfar_nfft = int(cfar_df["N_FFT"].values[0])
cfar_frames = int(cfar_df["Frames_Averaged"].values[0])

print(f"  [LOAD] Mixed signal  : {len(mixed_signal):,} samples")
print(f"  [LOAD] CFAR region   : centre {center_freq} kHz | BW {cfar_bw} Hz | "
      f"full-FFT bins [{k_low_full} -> {k_high_full}]")

if NORMAL_NOISE_STD is not None:
    rng = np.random.default_rng(0)
    normal_signal = normal_signal + rng.normal(0, NORMAL_NOISE_STD, len(normal_signal))
    print(f"  [LOAD] Added white noise (std={NORMAL_NOISE_STD}) to normal signal")


# =========================================
# STEP 2: BAND ISOLATION (same function for both classes)
# =========================================
def bandpass(x, lo, hi):
    """Keep full-FFT bins lo..hi (and their mirror); zero the rest."""
    n = len(x)
    X = np.fft.fft(x - np.mean(x))
    Y = np.zeros(n, dtype=complex)
    Y[lo:hi + 1] = X[lo:hi + 1]
    neg_lo = max(lo, 1)                       # do not double-count DC
    Y[n - hi:n - neg_lo + 1] = X[n - hi:n - neg_lo + 1]
    return np.real(np.fft.ifft(Y))


def strongest_band(x, n_fft, n_frames, drop_db=BW_DROP_DB):
    """
    Own strongest region of a record, found with the SAME spectral resolution
    as cacfar.py (n_fft-point averaged spectrum, peak + walk down drop_db) and
    converted to full-FFT bins the same way. This keeps the normal-class band
    as wide as the malicious-class band (same amount of in-band noise).
    """
    n = len(x)
    x = x - np.mean(x)
    n_avail = n // n_fft
    k = max(1, min(n_frames, n_avail))
    starts = np.unique(np.linspace(0, n_avail - 1, k).astype(int)) * n_fft
    win = np.hanning(n_fft)
    P = np.zeros(n_fft // 2 + 1)
    for s0 in starts:
        P += np.abs(np.fft.rfft(x[s0:s0 + n_fft] * win)) ** 2
    P_db = 10 * np.log10(P / len(starts) + 1e-18)
    pk = DC_IGNORE + int(np.argmax(P_db[DC_IGNORE:]))
    lo, hi = pk, pk
    while lo > 0 and P_db[lo - 1] > P_db[pk] - drop_db:
        lo -= 1
    while hi < len(P_db) - 1 and P_db[hi + 1] > P_db[pk] - drop_db:
        hi += 1
    scale = n / n_fft
    return int(np.floor(lo * scale)), min(int(np.ceil((hi + 1) * scale)) - 1, n // 2)


# malicious class: CFAR band of the mixed record
isolated_mal = bandpass(mixed_signal, k_low_full, k_high_full)

# normal class: its OWN strongest band, same isolation function
n_lo, n_hi = strongest_band(normal_signal, cfar_nfft, cfar_frames)
isolated_nor = bandpass(normal_signal, n_lo, n_hi)

print(f"  [ISO]  Malicious band bins [{k_low_full} -> {k_high_full}] | "
      f"normal band bins [{n_lo} -> {n_hi}]")


# =========================================
# STEP 3: FEATURES
# =========================================
def window_spectrum_stats(w, fs=FS):
    """
    Bandwidth and peak-above-noise MEASURED from one window of the
    unfiltered record. Same function for both classes -> no label encoding.
    """
    n = len(w)
    P = np.abs(np.fft.rfft((w - np.mean(w)) * np.hanning(n))) ** 2
    P_db = 10 * np.log10(P + 1e-18)
    noise_db = np.median(P_db)
    pk = DC_IGNORE + int(np.argmax(P_db[DC_IGNORE:]))
    lo, hi = pk, pk
    while lo > 0 and P_db[lo - 1] > P_db[pk] - BW_DROP_DB:
        lo -= 1
    while hi < len(P_db) - 1 and P_db[hi + 1] > P_db[pk] - BW_DROP_DB:
        hi += 1
    bw_hz = (hi - lo + 1) * fs / n
    return bw_hz, P_db[pk] - noise_db


def extract_features(iso_win, raw_win, label, block):
    """
    iso_win : band-isolated window  -> PAPR, variance, flatness, kurtosis
              (shape of the candidate waveform)
    raw_win : unfiltered window     -> bandwidth, Peak_to_Noise_Level_dB
              (need the out-of-band spectrum as noise reference, which the
               band-pass removes: in iso_win the median spectrum is ~0)

    Peak_to_Noise_Level_dB = (peak bin power) - (median bin power), in dB.
    It is NOT a strict SNR; do not call it SNR in the paper.
    """
    power = iso_win ** 2

    # 1. PAPR (dB)
    papr_db = 10 * np.log10(np.max(power) / (np.mean(power) + 1e-12))

    # 2. Variance
    variance = np.var(iso_win)

    # 3. Bandwidth (Hz) + 6. peak above noise (dB): measured per window
    bw, peak_to_noise_db = window_spectrum_stats(raw_win)

    # 4. Spectral flatness (geometric / arithmetic mean of magnitude spectrum)
    mag = np.abs(np.fft.fft(iso_win)[:len(iso_win) // 2]) + 1e-12
    flatness = np.exp(np.mean(np.log(mag))) / np.mean(mag)

    # 5. Kurtosis
    std = np.std(iso_win) + 1e-12
    kurt = np.mean(((iso_win - np.mean(iso_win)) / std) ** 4)

    return {
        "PAPR_dB": round(papr_db, 6),
        "Variance": round(variance, 6),
        "Bandwidth_Hz": round(bw, 4),
        "Spectral_Flatness": round(flatness, 6),
        "Kurtosis": round(kurt, 6),
        "Peak_to_Noise_Level_dB": round(peak_to_noise_db, 4),
        "Block_ID": block,
        "Label": label,
    }


# =========================================
# STEP 4: BUILD DATASET (same procedure for both classes)
# =========================================
n_win = min(len(mixed_signal), len(normal_signal)) // WINDOW_SIZE
rows = []
for i in range(n_win):
    sl = slice(i * WINDOW_SIZE, (i + 1) * WINDOW_SIZE)
    block = min(i * BLOCKS // n_win, BLOCKS - 1)
    rows.append(extract_features(isolated_mal[sl], mixed_signal[sl], 1, block))
    rows.append(extract_features(isolated_nor[sl], normal_signal[sl], 0, block))

features_df = pd.DataFrame(rows)
malicious_df = features_df[features_df["Label"] == 1]
normal_df = features_df[features_df["Label"] == 0]

print(f"\n  [FEAT] windows per class: {n_win}  | total rows: {len(features_df)}")

# --- leakage check: a feature that is (almost) constant inside each class
#     is probably encoding the label instead of measuring the signal.
feat_cols = ["PAPR_dB", "Variance", "Bandwidth_Hz",
             "Spectral_Flatness", "Kurtosis", "Peak_to_Noise_Level_dB"]
print("\n  [CHECK] within-class spread (std) per feature:")
for c in feat_cols:
    s1, s0 = malicious_df[c].std(), normal_df[c].std()
    flag = "   <-- near-constant, check this feature" if (s1 < 1e-9 or s0 < 1e-9) else ""
    print(f"     {c:<18} malicious {s1:12.5f} | normal {s0:12.5f}{flag}")

# =========================================
# STEP 5: PLOTS
# =========================================
titles = [
    "PAPR (dB)\nHigh = jammer spike",
    "Variance\nHigh = amplitude spread",
    "Bandwidth (Hz)\nMeasured per window",
    "Spectral Flatness\nHigh = noise-like",
    "Kurtosis\nHigh = impulsive jammer",
    "Peak-to-noise level (dB)\nMeasured per window",
]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Feature Extraction - Malicious vs Normal", fontsize=13, fontweight="bold")
for idx, (feat, title) in enumerate(zip(feat_cols, titles)):
    ax = axes[idx // 3][idx % 3]
    ax.hist(malicious_df[feat], bins=20, alpha=0.7, color="red",
            label="Malicious (1)", edgecolor="darkred")
    ax.hist(normal_df[feat], bins=20, alpha=0.7, color="blue",
            label="Normal (0)", edgecolor="darkblue")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(feat)
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(path("feature_extraction_output.png"), dpi=150, bbox_inches="tight")
if SHOW_PLOT:
    plt.show()
print("\n  [PLOT] Saved: feature_extraction_output.png")

# =========================================
# STEP 6: EXPORT
# =========================================
features_df.to_csv(path("features_dataset.csv"), index=False)
print(f"  [CSV]  Saved: features_dataset.csv  ({len(features_df)} rows)")
print(f"         Columns: {list(features_df.columns)}")
print("         In ml_classifier.py: drop Block_ID from X and split by Block_ID")
print("         (e.g. GroupShuffleSplit / GroupKFold), not by random rows.")
print("\n" + "=" * 55)
print("  Feature Extraction Complete  -> next: ml_classifier.py")
print("=" * 55)