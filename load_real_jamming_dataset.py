"""
load_real_jamming_dataset.py

Extracts features from the REAL RF Jamming Spectral Scan dataset
(Kaggle: daniaherzalla/radio-frequency-jamming). Unlike RadioML, this
dataset is TIME-SERIES TELEMETRY (rssi, power readings over ~30000 samples
per file), not raw I/Q waveforms - so features are computed differently,
and 2 of your original features (Order2_PAR, Order4_PAR - replacing the
old Phase_State_Count/Phase_Variance discriminator) CANNOT be computed
from this data at all (no modulated carrier / I-Q phase info available,
only a scalar RSSI/power reading) - these are set to 0 for all rows from
this source, same placeholder role the old Phase_State_Count/Phase_Variance
played.

Each CSV file = ONE signal capture (30000 time-series samples).
We treat the 'rssi' and 'avgpwr_db' columns as the "signal" to extract
features from, since those represent the actual received signal strength
over time - closest real-world equivalent to your synthetic time-domain
signal.

Usage:
    Put this script in the folder containing your downloaded sample
    files, update SAMPLE_FILES below with your actual file paths, then:
    python load_real_jamming_dataset.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kurtosis, skew
from scipy.signal import welch, find_peaks

# ---------------------------------------------------------
# Config - UPDATE THESE PATHS to your actual downloaded files
# ---------------------------------------------------------
# ---------------------------------------------------------
# Config - point at FOLDERS instead of listing individual files.
# The script scans every .csv inside each folder (recursively) and
# assigns the given Label/Label_Name to all of them.
# ---------------------------------------------------------
SOURCE_FOLDERS = [
    # (folder_path, Label, Label_Name, max_files_to_use)
    # max_files = None means process EVERY .csv file found in that folder
    (r"D:\projectsml\active_scan\malicious\singletone\0dbm", 3, "Jam_Tone", None),
    (r"D:\projectsml\active_scan\malicious\gaussian_noise\0dbm", 7, "Jam_Gaussian", None),
    (r"D:\projectsml\active_scan\benign\background\location1", 0, "Normal_Background", None),
    # Add more rows here for other power levels or benign locations, e.g.:
    # (r"C:\Users\ashwi\Downloads\active_scan\malicious\singletone\10dbm", 3, "Jam_Tone", None),
    # (r"C:\Users\ashwi\Downloads\active_scan\benign\floor", 0, "Normal_Background", None),
]

FS = 1.0   # This telemetry has no fixed sample rate given - treat as
           # unitless sample index; spectral features are still valid
           # relative comparisons even without a true Hz-based Fs.


def extract_features_from_telemetry(df):
    """
    Extract a 13-feature vector (2 will be zero - see module docstring)
    from a single spectral-scan CSV's time-series columns.
    """
    # Use 'rssi' as the primary time-series signal (received power over
    # time - closest real-world analog to a raw signal's envelope)
    signal = df['rssi'].values.astype(float)
    n = len(signal)

    # PAPR (based on rssi as a proxy for signal power, not true I/Q power)
    peak_power = np.max(np.abs(signal)) ** 2
    avg_power = np.mean(signal ** 2) + 1e-12
    papr_db = 10 * np.log10(peak_power / avg_power + 1e-12)

    variance = np.var(signal)

    freqs, psd = welch(signal, fs=FS, nperseg=min(256, n))
    psd = psd + 1e-12
    gm = np.exp(np.mean(np.log(psd)))
    am = np.mean(psd)
    spectral_flatness = gm / am

    kurt = kurtosis(signal)

    psd_norm = psd / np.sum(psd)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    spectral_entropy /= np.log2(len(psd_norm))

    skewness = skew(signal)

    zero_crossings = np.sum(np.abs(np.diff(np.sign(signal - np.mean(signal)))) > 0)
    zcr = zero_crossings / (n - 1)

    threshold = np.max(psd) * 0.3
    peaks, _ = find_peaks(psd, height=threshold)
    spectral_peak_count = len(peaks)

    envelope = np.abs(signal - np.mean(signal))
    window = 20
    kernel = np.ones(window) / window
    smooth_env = np.convolve(envelope, kernel, mode="same")
    env_threshold = np.max(smooth_env) * 0.5
    duty_cycle = np.mean(smooth_env >= env_threshold)

    sig_zero_mean = signal - np.mean(signal)
    zero_lag_energy = np.sum(sig_zero_mean ** 2) + 1e-12
    max_lag = min(500, n // 2)
    autocorr_vals = []
    for lag in range(50, max_lag, 10):   # step by 10 - 30000 samples is a lot, skip for speed
        ac = np.sum(sig_zero_mean[:-lag] * sig_zero_mean[lag:])
        autocorr_vals.append(abs(ac) / zero_lag_energy)
    echo_autocorr_strength = max(autocorr_vals) if autocorr_vals else 0.0

    # NOT computable from this data source - RSSI/power telemetry has no
    # I/Q or carrier phase info, so the M-th power nonlinearity technique
    # (used in load_radioml_dataset.py / generate_multiclass_dataset.py to
    # fix BPSK/QPSK confusion) has nothing to operate on here. Zeroed out,
    # same placeholder role the old Phase_State_Count/Phase_Variance played.
    order2_par = 0.0
    order4_par = 0.0
    phase_variance = 0.0

    # BONUS feature unique to this real dataset: frequency variation.
    # Jammers observed sitting on a fairly fixed channel but with some
    # sweep; legitimate background scanning hops across channels more.
    # Not part of your original 12 - stored separately, optional to use.
    freq_variation = df['freq1'].std() if 'freq1' in df.columns else 0.0

    return {
        "PAPR_dB": papr_db,
        "Variance": variance,
        "Spectral_Flatness": spectral_flatness,
        "Kurtosis": kurt,
        "Spectral_Entropy": spectral_entropy,
        "Skewness": skewness,
        "ZCR": zcr,
        "Spectral_Peak_Count": spectral_peak_count,
        "Envelope_Duty_Cycle": duty_cycle,
        "Echo_Autocorr_Strength": echo_autocorr_strength,
        "Order2_PAR": order2_par,
        "Order4_PAR": order4_par,
        "Phase_Variance": phase_variance,
        "Freq_Variation_Hz": freq_variation,   # extra, optional
    }


def main():
    print("=" * 60)
    print("  Extracting features from REAL RF jamming telemetry data")
    print("=" * 60)

    rows = []
    for folder_path, label, label_name, max_files in SOURCE_FOLDERS:
        folder = Path(folder_path)
        if not folder.exists():
            print(f"  [SKIP] Folder not found: {folder_path}")
            continue

        # Recursively find all CSVs under this folder (handles nested
        # subfolders like location1/location2 automatically)
        csv_files = sorted(folder.rglob("*.csv"))
        if not csv_files:
            print(f"  [SKIP] No .csv files found under: {folder_path}")
            continue

        csv_files = csv_files if max_files is None else csv_files[:max_files]
        print(f"\n  [FOLDER] {folder_path}")
        cap_msg = "ALL files" if max_files is None else f"up to {max_files}"
        print(f"           Found {len(csv_files)} files (using {cap_msg}) -> {label_name}")

        for i, filepath in enumerate(csv_files):
            try:
                df = pd.read_csv(filepath)
                feats = extract_features_from_telemetry(df)
                feats["Label"] = label
                feats["Label_Name"] = label_name
                feats["Source"] = "real_jamming_telemetry"
                feats["Source_File"] = filepath.name
                rows.append(feats)
                if (i + 1) % 10 == 0 or (i + 1) == len(csv_files):
                    print(f"           [{i+1}/{len(csv_files)}] processed: {filepath.name}")
                    # Periodic save every 10 files, in case a huge folder
                    # takes a long time - you won't lose all progress if
                    # it gets interrupted partway through
                    pd.DataFrame(rows).to_csv("real_jamming_features.csv", index=False)
            except Exception as e:
                print(f"           [ERROR] {filepath.name}: {e}")

    if not rows:
        print("\n  No files were successfully loaded. Check SOURCE_FOLDERS paths.")
        return

    result_df = pd.DataFrame(rows)
    result_df.to_csv("real_jamming_features.csv", index=False)
    print(f"\n{'='*60}")
    print(f"Saved {len(result_df)} rows to real_jamming_features.csv")
    print(f"\nClass breakdown:")
    print(result_df["Label_Name"].value_counts())
    print(f"\nFeature means by class:")
    feature_cols = ["PAPR_dB", "Variance", "Spectral_Flatness", "Kurtosis",
                     "Spectral_Entropy", "Skewness", "ZCR",
                     "Spectral_Peak_Count", "Envelope_Duty_Cycle",
                     "Echo_Autocorr_Strength", "Freq_Variation_Hz"]
    print(result_df.groupby("Label_Name")[feature_cols].mean().round(3))


if __name__ == "__main__":
    main()