# =========================================
# CA-CFAR Detection — Fixed Version
# Project: Malicious RF Signal Detection
# Input  : mixed_rf_signal.csv
# Output : cfar_output.csv
# =========================================
"""
CA-CFAR Malicious Signal Detection  (v2)
Project : Malicious RF Signal Detection
Input   : mixed_rf_signal.csv   (columns: Time_sec, Mixed_Signal)
Output  : cfar_output.csv       (primary region, used by feature_extraction.py)
          cfar_regions.csv      (every CFAR region found)
          cfar_detection_output.png

What changed vs v1
  1. CFAR is actually used. v1 computed detections and then ignored them
     (it took np.argmax of the spectrum). Now the detected CFAR clusters
     decide the malicious region.
  2. CFAR runs on LINEAR POWER (not dB), threshold = alpha * mean(ref cells).
     v1 added alpha in dB to a dB-mean, which is not CA-CFAR.
  3. alpha is exact for frame-averaged spectra (F-distribution), so the
     real Pfa matches the Pfa you set. Reduces to N*(Pfa^(-1/N)-1) for 1 frame.
  4. Defaults match paper Table II: N_FFT=1024, 16 ref, 4 guard, Pfa=1e-4,
     10-frame averaging. Everything is a constant you can change below.
  5. Adjacent detected bins are merged into regions; several regions are
     reported; the strongest one is the primary region.
  6. DC offset removed, Hann window used, DC bins ignored.
  7. NO fallback: if CFAR detects nothing the script says so, writes
     Detection=NO_DETECTION and exits with code 2 (v1 used argmax anyway).
  8. No hard-coded E:\\ path, no hard-coded "433.92 MHz / JSR / SNR" labels.
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import f as f_dist
except ImportError:          # scipy missing -> fall back to single-frame formula
    f_dist = None

# =========================================
# CONFIG  (edit here)
# =========================================
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:            # running inside a notebook
    _HERE = os.getcwd()
DATA_DIR = os.environ.get("RF_DATA_DIR", _HERE)      # folder with the CSV files

INPUT_CSV  = "mixed_rf_signal.csv"
FS         = 2.048e6          # sampling rate (Hz)

N_FFT      = 1024             # paper Table II
FRAME_AVERAGING = True        # paper: used for low SNR (< -12 dB)
N_AVG_FRAMES    = 10          # paper: N = 10 frames

N_REF      = 16               # reference cells EACH side of CUT (paper)
N_GUARD    = 4                # guard cells EACH side of CUT (paper)
PFA        = 1e-4             # target false-alarm probability (paper)

DC_IGNORE_BINS     = 3        # ignore bins next to 0 Hz
MIN_ABOVE_NOISE_DB = 0.0      # optional extra power floor over median noise
                              # (v1 used 25 dB). 0 = off -> pure CFAR.
BW_DROP_DB         = 10.0     # bandwidth = region where power > peak - 10 dB
MERGE_GAP_BINS     = N_GUARD  # detected bins this close are one region

CENTER_RF_HZ = None           # e.g. 433.92e6 to also print the RF frequency
SHOW_PLOT    = True


CSV_HEADER = [
    "Center_Freq_kHz", "Bandwidth_Hz",
    "Peak_Power_dB", "Noise_Floor_dB",
    "SNR_above_noise_dB", "Peak_Freq_kHz",
    "Bin_Low", "Bin_High", "Detection",
    # new columns (feature_extraction.py should use the *_Full ones)
    "Bin_Low_Full", "Bin_High_Full", "N_FFT",
    "Frames_Averaged", "Alpha", "Num_Regions",
]
REGION_HEADER = ["Rank", "Bin_Low", "Bin_High", "Peak_Freq_kHz", "Peak_Power_dB"]


def path(name):
    return os.path.join(DATA_DIR, name)


print("=" * 55)
print("  CA-CFAR Malicious Signal Detection (v2)")
print("=" * 55)

# =========================================
# STEP 1: LOAD SIGNAL
# =========================================
df = pd.read_csv(path(INPUT_CSV))
mixed_signal = df["Mixed_Signal"].values.astype(float)
t = df["Time_sec"].values
N = len(mixed_signal)
print(f"  [LOAD] {N:,} samples @ fs = {FS/1e6:.3f} MHz")

# remove DC offset (paper: DC spike removal)
x = mixed_signal - np.mean(mixed_signal)

# =========================================
# STEP 2: (AVERAGED) POWER SPECTRUM
# =========================================
n_frames_avail = N // N_FFT
if n_frames_avail < 1:
    raise ValueError(f"Signal shorter than one FFT frame ({N} < {N_FFT})")

K = min(N_AVG_FRAMES, n_frames_avail) if FRAME_AVERAGING else 1
# frames spread over the whole record, so pulsed jammers are not missed
starts = np.unique(np.linspace(0, n_frames_avail - 1, K).astype(int)) * N_FFT
K = len(starts)

win = np.hanning(N_FFT)
win_pow = np.sum(win ** 2)
P = np.zeros(N_FFT // 2 + 1)
for s in starts:
    seg = x[s:s + N_FFT] * win
    P += np.abs(np.fft.rfft(seg)) ** 2 / win_pow
P /= K                                              # linear power per bin

P_db = 10 * np.log10(P + 1e-18)
frequencies = np.fft.rfftfreq(N_FFT, 1 / FS)
freq_res = FS / N_FFT

print(f"  [FFT]  N_FFT = {N_FFT}, frames averaged = {K}, "
      f"resolution = {freq_res:.2f} Hz/bin")

# =========================================
# STEP 3: CA-CFAR  (linear power)
# =========================================
n_ref_total = 2 * N_REF


def cfar_alpha(n_ref, k_frames, pfa):
    """
    Threshold multiplier: detect when P_cut > alpha * mean(reference cells).
    With K averaged frames, P_cut/mean_ref ~ F(2K, 2K*n_ref) under noise only,
    so alpha is the upper-tail quantile. For K=1: n_ref*(pfa^(-1/n_ref) - 1).
    """
    if f_dist is not None:
        return float(f_dist.isf(pfa, 2 * k_frames, 2 * k_frames * n_ref))
    if k_frames > 1:
        print("  [WARN] scipy not found: using single-frame alpha "
              "(conservative: real Pfa will be lower than set)")
    return float(n_ref * (pfa ** (-1.0 / n_ref) - 1))


alpha = cfar_alpha(n_ref_total, K, PFA)

half = N_REF + N_GUARD
kernel = np.zeros(2 * half + 1)
kernel[:N_REF] = 1.0                 # left reference cells
kernel[-N_REF:] = 1.0                # right reference cells (guard + CUT = 0)
noise_est = np.convolve(P, kernel, mode="same") / n_ref_total
threshold = alpha * noise_est

valid = np.zeros(len(P), dtype=bool)
valid[half:len(P) - half] = True      # only cells with a full window
valid[:DC_IGNORE_BINS] = False

detections = valid & (P > threshold)

noise_floor_db = float(np.median(P_db))          # global noise floor estimate
if MIN_ABOVE_NOISE_DB > 0:
    detections &= (P_db > noise_floor_db + MIN_ABOVE_NOISE_DB)

print(f"\n  [CFAR] ref = {N_REF}/side, guard = {N_GUARD}/side, "
      f"Pfa = {PFA:g}, alpha = {alpha:.3f} ({10*np.log10(alpha):.2f} dB)")
print(f"         Noise floor (median): {noise_floor_db:.2f} dB")
print(f"         Bins above CFAR threshold: {int(detections.sum())}")

# =========================================
# STEP 4: MERGE DETECTED BINS INTO REGIONS
# =========================================
idx = np.where(detections)[0]
regions = []
if len(idx):
    splits = np.where(np.diff(idx) > MERGE_GAP_BINS)[0] + 1
    for grp in np.split(idx, splits):
        lo, hi = int(grp[0]), int(grp[-1])
        pk = lo + int(np.argmax(P[lo:hi + 1]))
        regions.append({"lo": lo, "hi": hi, "peak": pk, "peak_db": float(P_db[pk])})
    regions.sort(key=lambda r: r["peak_db"], reverse=True)

if not regions:
    # CFAR found nothing -> report "no detection". No region is invented.
    print("\n  [RESULT] NO MALICIOUS SIGNAL DETECTED")
    print("           (no bin crossed the CA-CFAR threshold)")

    thr_db = np.where(valid, 10 * np.log10(threshold + 1e-18), np.nan)
    plt.figure(figsize=(14, 4))
    plt.plot(frequencies / 1e3, P_db, color="black", lw=0.6, label="Power spectrum")
    plt.plot(frequencies / 1e3, thr_db, color="orange", lw=1.4, ls="--",
             label=f"CA-CFAR threshold (Pfa={PFA:g})")
    plt.title("CA-CFAR: no detection")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Power (dB)")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path("cfar_detection_output.png"), dpi=150, bbox_inches="tight")
    if SHOW_PLOT:
        plt.show()

    with open(path("cfar_output.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerow([""] * 8 + ["NO_DETECTION"] + [""] * 2 + [N_FFT, K, round(alpha, 4), 0])
    with open(path("cfar_regions.csv"), "w", newline="") as f:
        csv.writer(f).writerow(REGION_HEADER)

    print("  [CSV]  cfar_output.csv written with Detection = NO_DETECTION")
    print("         feature_extraction.py must skip this record.")
    sys.exit(2)          # non-zero exit code so a batch run can stop here

print(f"\n  [REGIONS] {len(regions)} region(s) (strongest first):")
for i, r in enumerate(regions[:5]):
    print(f"     #{i+1}: bins {r['lo']}-{r['hi']}  "
          f"peak {frequencies[r['peak']]/1e3:.3f} kHz  {r['peak_db']:.2f} dB")

# =========================================
# STEP 5: PRIMARY REGION -> centre, bandwidth
# =========================================
main = regions[0]
peak_bin = main["peak"]
peak_power_db = main["peak_db"]
peak_freq_hz = float(frequencies[peak_bin])

# walk down from the peak until power falls BW_DROP_DB below it
bw_level = peak_power_db - BW_DROP_DB
k_low, k_high = peak_bin, peak_bin
while k_low > 0 and P_db[k_low - 1] > bw_level:
    k_low -= 1
while k_high < len(P_db) - 1 and P_db[k_high + 1] > bw_level:
    k_high += 1
# the CFAR region itself is also part of the signal footprint
k_low = min(k_low, main["lo"])
k_high = max(k_high, main["hi"])

center_freq_hz = float(frequencies[(k_low + k_high) // 2])
bandwidth_hz = (k_high - k_low + 1) * freq_res
snr_above_noise = peak_power_db - noise_floor_db

print("\n  [RESULT] Malicious signal detected")
print(f"  {'-'*45}")
print(f"  Peak Frequency   : {peak_freq_hz/1e3:.4f} kHz (baseband)")
if CENTER_RF_HZ is not None:
    print(f"                     -> {(CENTER_RF_HZ + peak_freq_hz)/1e6:.4f} MHz (RF)")
print(f"  Center Frequency : {center_freq_hz/1e3:.4f} kHz")
print(f"  Bandwidth        : {bandwidth_hz:.2f} Hz  "
      f"({k_high-k_low+1} bin(s) of {freq_res:.1f} Hz)")
print(f"  Peak Power       : {peak_power_db:.2f} dB")
print(f"  Noise Floor      : {noise_floor_db:.2f} dB")
print(f"  Peak above noise : {snr_above_noise:.2f} dB")
print(f"  Bin Range        : [{k_low} -> {k_high}]")
print(f"  {'-'*45}")

# =========================================
# STEP 6: ISOLATE MALICIOUS WAVEFORM
# =========================================
scale = N / N_FFT                                   # N_FFT-bin -> full-FFT bin
k_low_full = int(np.floor(k_low * scale))
k_high_full = min(int(np.ceil((k_high + 1) * scale)) - 1, N // 2)

fft_full = np.fft.fft(x)
fft_isolated = np.zeros(N, dtype=complex)
fft_isolated[k_low_full:k_high_full + 1] = fft_full[k_low_full:k_high_full + 1]
if k_low_full == 0:
    neg_lo = 1                                       # do not double-count DC
else:
    neg_lo = k_low_full
fft_isolated[N - k_high_full:N - neg_lo + 1] = fft_full[N - k_high_full:N - neg_lo + 1]
isolated_waveform = np.real(np.fft.ifft(fft_isolated))

print(f"  [ISO]  Isolated {len(isolated_waveform):,} samples, "
      f"full-FFT bins [{k_low_full} -> {k_high_full}]")
print(f"         Amplitude range: [{isolated_waveform.min():.4f}, "
      f"{isolated_waveform.max():.4f}]")

# =========================================
# STEP 7: PLOTS
# =========================================
thr_db = np.where(valid, 10 * np.log10(threshold + 1e-18), np.nan)
f_khz = frequencies / 1e3

fig, axes = plt.subplots(3, 1, figsize=(14, 11))
fig.suptitle("CA-CFAR Malicious Signal Detection", fontsize=13, fontweight="bold")

axes[0].plot(f_khz, P_db, color="black", lw=0.6,
             label=f"Power spectrum ({K}-frame average)")
axes[0].plot(f_khz, thr_db, color="orange", lw=1.4, ls="--",
             label=f"CA-CFAR threshold (Pfa={PFA:g})")
axes[0].plot(f_khz[detections], P_db[detections], "r.", ms=5, label="CFAR detections")
axes[0].axhline(noise_floor_db, color="green", ls=":", lw=1.2,
                label=f"Noise floor {noise_floor_db:.1f} dB")
axes[0].set_title("Panel 1: Spectrum with adaptive CA-CFAR threshold")
axes[0].set_xlabel("Frequency (kHz)")
axes[0].set_ylabel("Power (dB)")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(f_khz, P_db, color="black", lw=0.6, label="Spectrum")
for i, r in enumerate(regions):
    axes[1].axvspan(frequencies[r["lo"]] / 1e3, frequencies[r["hi"]] / 1e3,
                    alpha=0.25, color="blue" if i else "red",
                    label="Other CFAR region" if i == 1 else None)
axes[1].axvspan(frequencies[k_low] / 1e3, frequencies[k_high] / 1e3,
                alpha=0.4, color="red", label="Primary malicious region")
axes[1].axvline(center_freq_hz / 1e3, color="red", lw=1.5,
                label=f"Centre {center_freq_hz/1e3:.2f} kHz")
axes[1].set_title(f"Panel 2: Detection map | BW {bandwidth_hz:.1f} Hz | "
                  f"Peak {peak_power_db:.1f} dB")
axes[1].set_xlabel("Frequency (kHz)")
axes[1].set_ylabel("Power (dB)")
axes[1].legend(loc="upper right", fontsize=8)
axes[1].grid(True, alpha=0.3)

ns = min(1000, len(isolated_waveform))
axes[2].plot(t[:ns] * 1e3, isolated_waveform[:ns], color="red", lw=0.8,
             label="Isolated malicious waveform")
axes[2].set_title("Panel 3: Isolated malicious waveform")
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("Amplitude")
axes[2].legend(loc="upper right", fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(path("cfar_detection_output.png"), dpi=150, bbox_inches="tight")
if SHOW_PLOT:
    plt.show()
print("\n  [PLOT] Saved: cfar_detection_output.png")

# =========================================
# STEP 8: EXPORT
# =========================================
with open(path("cfar_output.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(CSV_HEADER)
    w.writerow([
        round(center_freq_hz / 1e3, 4), round(bandwidth_hz, 4),
        round(peak_power_db, 4), round(noise_floor_db, 4),
        round(snr_above_noise, 4), round(peak_freq_hz / 1e3, 4),
        k_low, k_high,
        "MALICIOUS",
        k_low_full, k_high_full, N_FFT, K, round(alpha, 4), len(regions),
    ])

with open(path("cfar_regions.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(REGION_HEADER)
    for i, r in enumerate(regions, 1):
        w.writerow([i, r["lo"], r["hi"],
                    round(frequencies[r["peak"]] / 1e3, 4), round(r["peak_db"], 4)])

print("  [CSV]  Saved: cfar_output.csv, cfar_regions.csv")
print("\n" + "=" * 55)
print("  CA-CFAR Complete - Ready for feature_extraction.py")
print("=" * 55)