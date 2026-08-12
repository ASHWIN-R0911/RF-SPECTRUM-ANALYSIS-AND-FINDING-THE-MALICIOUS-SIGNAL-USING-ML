"""
load_radioml_dataset.py

Loads the REAL RadioML2016.10a dataset (downloaded from deepsig.ai/datasets)
and extracts the same 12 features used in generate_multiclass_dataset.py,
for the 3 classes matching your project: BPSK, QPSK, WBFM (used as your FM
class).

RadioML2016.10a format: a Python pickle containing a dict where:
    key   = (modulation_name, SNR_dB)   e.g. ('BPSK', 10)
    value = numpy array of shape (1000, 2, 128)
            1000 signal instances, each with 2 rows (I and Q), 128 samples

Since RadioML gives you the actual I and Q channels directly (not a
real-valued mixed signal), we can reconstruct a real bandpass-style signal
by treating I/Q as a complex baseband signal, OR extract features straight
from the complex I/Q data (more accurate - this is what real receivers do).
This script does the latter: features are computed from the complex I+jQ
signal directly, using a version of extract_features adapted for complex
input.

Run:
    python load_radioml_dataset.py
Requires:
    RML2016.10a_dict.pkl in the same folder (or update PKL_PATH below)
Output:
    real_signals_dataset.csv
"""

import pickle
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import welch, find_peaks

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
PKL_PATH = "RML2016.10a_dict.pkl"     # update path if needed
FS = 1_000_000                          # RadioML doesn't specify exact Fs; use consistent value for feature math
CARRIER_FREQ_ASSUMED = 0                # RadioML I/Q is already baseband (no carrier to demodulate)

# Map RadioML's class names -> your project's class names
CLASS_MAP = {
    "BPSK": ("BPSK", 0),
    "QPSK": ("QPSK", 1),
    "WBFM": ("FM", 2),
}

# Use a moderate SNR range - very low SNR signals are unrealistically hard
# and would hurt your accuracy; very high SNR is unrealistically easy.
# 0 to 18 dB is a reasonable "usable signal" range.
SNR_RANGE = list(range(0, 19, 2))   # 0,2,4,...,18 dB
SAMPLES_PER_CLASS = 1000             # cap so file size / runtime stay reasonable


def extract_features_complex(iq_signal):
    """
    12-feature extraction adapted for complex baseband I/Q input
    (RadioML gives you this directly - no need to demodulate a carrier).
    iq_signal: complex numpy array of shape (n,)
    """
    real_sig = np.real(iq_signal)
    n = len(iq_signal)

    # PAPR (based on magnitude of complex signal)
    mag = np.abs(iq_signal)
    peak_power = np.max(mag) ** 2
    avg_power = np.mean(mag ** 2)
    papr_db = 10 * np.log10(peak_power / (avg_power + 1e-12))

    variance = np.var(real_sig)

    freqs, psd = welch(real_sig, fs=FS, nperseg=min(128, n))
    psd = psd + 1e-12
    gm = np.exp(np.mean(np.log(psd)))
    am = np.mean(psd)
    spectral_flatness = gm / am

    kurt = kurtosis(real_sig)

    psd_norm = psd / np.sum(psd)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    spectral_entropy /= np.log2(len(psd_norm))

    skewness = skew(real_sig)

    zero_crossings = np.sum(np.abs(np.diff(np.sign(real_sig))) > 0)
    zcr = zero_crossings / (n - 1)

    threshold = np.max(psd) * 0.3
    peaks, _ = find_peaks(psd, height=threshold)
    spectral_peak_count = len(peaks)

    envelope = mag
    window = min(8, n // 4) or 1
    kernel = np.ones(window) / window
    smooth_env = np.convolve(envelope, kernel, mode="same")
    env_threshold = np.max(smooth_env) * 0.5
    duty_cycle = np.mean(smooth_env >= env_threshold)

    sig_zero_mean = real_sig - np.mean(real_sig)
    zero_lag_energy = np.sum(sig_zero_mean ** 2) + 1e-12
    max_lag = min(50, n // 2)
    autocorr_vals = []
    for lag in range(5, max_lag):
        ac = np.sum(sig_zero_mean[:-lag] * sig_zero_mean[lag:])
        autocorr_vals.append(abs(ac) / zero_lag_energy)
    echo_autocorr_strength = max(autocorr_vals) if autocorr_vals else 0.0

    # --- M-th power nonlinearity features (replaces old Phase_State_Count) ---
    # Classic AMC (automatic modulation classification) technique: raising the
    # complex I/Q signal to the M-th power removes an M-PSK signal's phase
    # modulation, collapsing its spectrum to a single sharp tone.
    #   - BPSK (2 phase states): squaring (^2) already removes the modulation
    #     -> sharp tone at BOTH order-2 and order-4
    #   - QPSK (4 phase states): squaring leaves 2 residual phase states
    #     (still "modulated") -> spectrum stays spread at order-2; only
    #     collapses to a sharp tone at order-4
    # We measure "sharp tone vs spread spectrum" via peak-to-average power
    # ratio (PAR) of the FFT. This replaces the old phase-histogram peak
    # counting, which was noise-fragile and gave heavily overlapping BPSK/QPSK
    # distributions (mean 4.9 vs 7.5, ranges 1-10 vs 4-11 - nearly useless).
    def _mth_power_par(z, m):
        zm = z ** m
        Zf = np.abs(np.fft.fft(zm)) ** 2
        return np.max(Zf) / (np.mean(Zf) + 1e-12)

    order2_par = _mth_power_par(iq_signal, 2)
    order4_par = _mth_power_par(iq_signal, 4)

    phase = np.angle(iq_signal)
    phase_mod = np.mod(phase, 2 * np.pi)
    phase_variance = np.var(phase_mod)

    return [papr_db, variance, spectral_flatness, kurt, spectral_entropy,
            skewness, zcr, spectral_peak_count, duty_cycle,
            echo_autocorr_strength, order2_par, order4_par, phase_variance]


FEATURE_COLS = ["PAPR_dB", "Variance", "Spectral_Flatness", "Kurtosis",
                "Spectral_Entropy", "Skewness", "ZCR",
                "Spectral_Peak_Count", "Envelope_Duty_Cycle",
                "Echo_Autocorr_Strength", "Order2_PAR", "Order4_PAR", "Phase_Variance"]


def main():
    print("=" * 60)
    print("  Loading RadioML2016.10a (real-world signal dataset)")
    print("=" * 60)

    with open(PKL_PATH, "rb") as f:
        radioml_dict = pickle.load(f, encoding="latin1")

    rows = []
    for radioml_name, (project_name, label_id) in CLASS_MAP.items():
        count_this_class = 0
        # Spread SAMPLES_PER_CLASS evenly across every SNR bin instead of
        # filling up entirely from the first bin. RadioML gives exactly
        # 1000 samples per (mod, snr) key, and the old loop broke out of
        # the OUTER snr loop as soon as count_this_class hit
        # SAMPLES_PER_CLASS - since that happened while still inside the
        # very first SNR bin (0 dB), every class was trained/tested
        # exclusively on the noisiest, hardest-case signals and never saw
        # cleaner 2-18 dB examples at all. This directly hurt BPSK/QPSK
        # separation, since the M-th power PAR features (like any
        # feature) are noisier at 0 dB than at higher SNR.
        available_snrs = [s for s in SNR_RANGE if (radioml_name, s) in radioml_dict]
        per_bin_target = max(1, SAMPLES_PER_CLASS // max(1, len(available_snrs)))
        for snr in available_snrs:
            key = (radioml_name, snr)
            samples = radioml_dict[key]   # shape (1000, 2, 128)
            n_from_this_bin = 0
            for sample in samples:
                if n_from_this_bin >= per_bin_target or count_this_class >= SAMPLES_PER_CLASS:
                    break
                iq = sample[0] + 1j * sample[1]   # complex baseband signal
                feats = extract_features_complex(iq)
                row = dict(zip(FEATURE_COLS, feats))
                row["Label"] = label_id
                row["Label_Name"] = f"Normal_{project_name}" if project_name != "FM" else "Normal_FM"
                row["SNR_dB"] = snr
                row["Source"] = "RadioML2016.10a (real)"
                rows.append(row)
                count_this_class += 1
                n_from_this_bin += 1
            if count_this_class >= SAMPLES_PER_CLASS:
                break
        print(f"  [{radioml_name}] extracted {count_this_class} real samples "
              f"across {len(available_snrs)} SNR bins ({available_snrs})")

    df = pd.DataFrame(rows)
    df.to_csv("real_signals_dataset.csv", index=False)
    print(f"\nSaved {len(df)} rows to real_signals_dataset.csv")
    print(df.groupby("Label_Name")[FEATURE_COLS].mean().round(3))


if __name__ == "__main__":
    main()