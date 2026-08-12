# =========================================
# classify_live.py
# Interactive LIVE classification demo for evaluation.
#
# Lets you pick a signal type on the spot, generates it, runs it through
# your trained SVM+RF ensemble, and prints the classification result
# immediately - including full per-class confidence breakdown.
#
# IMPORTANT: this project's 8 classes were trained from TWO different
# pipelines that must be replicated exactly, or a live-generated demo
# signal will look "out of distribution" to the model and misclassify:
#   - BPSK / QPSK / FM   -> trained EXCLUSIVELY on real RadioML2016.10a
#                           data: short complex I/Q sequences (~128
#                           samples), extracted via extract_features_complex
#                           (load_radioml_dataset.py). merge_final_hybrid_
#                           dataset.py discards the synthetic generator's
#                           BPSK/QPSK/FM entirely.
#   - Tone/Sweep/Pulsed/Gaussian-Jam, Spoofed-Replay
#                        -> trained on real Kaggle telemetry (Tone/
#                           Gaussian) or the synthetic generator's
#                           real-valued 1024-sample carrier signals
#                           (Sweep/Pulsed/Spoofed), extracted via
#                           extract_features (generate_multiclass_dataset.py)
#
# This script uses the matching generator + extractor pair for each
# class so every demo signal is statistically representative of what
# the model actually learned.
# =========================================

import os
os.chdir(r'D:\projectsml')

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.signal import welch, find_peaks, resample

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
FS = 1_000_000              # synthetic-generator sample rate (Sweep/Pulsed/Spoofed)
N_SAMPLES = 1024             # synthetic-generator window length
CARRIER_FREQ = 204_800       # synthetic-generator carrier

RADIOML_FS = 1_000_000       # load_radioml_dataset.py's assumed Fs for spectral math
rng = np.random.default_rng()

FEATURE_COLS = ['PAPR_dB', 'Variance', 'Spectral_Flatness', 'Kurtosis',
                'Spectral_Entropy', 'Skewness', 'ZCR',
                'Spectral_Peak_Count', 'Envelope_Duty_Cycle',
                'Echo_Autocorr_Strength', 'Order2_PAR', 'Order4_PAR', 'Phase_Variance']

LABEL_NAMES = {0: 'BPSK', 1: 'QPSK', 2: 'FM',
               3: 'Tone-Jam', 4: 'Sweep-Jam', 5: 'Pulsed-Jam',
               6: 'Spoofed-Replay', 7: 'Gaussian-Jam'}
MALICIOUS_LABELS = {3, 4, 5, 6, 7}


# =====================================================================
# PATHWAY A: RadioML-style complex I/Q generators, for clean BPSK/QPSK/FM
# (matches load_radioml_dataset.py's actual real training data)
# =====================================================================
def make_bpsk_iq(n_symbols=32, sps=4, snr_db=None):
    if snr_db is None:
        snr_db = rng.uniform(0, 18)   # same usable SNR range as training
    bits = rng.integers(0, 2, n_symbols)
    symbols = np.exp(1j * np.pi * bits)
    sig = np.repeat(symbols, sps)
    n = len(sig)
    p = np.mean(np.abs(sig) ** 2)
    noise_p = p / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_p / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return sig + noise


def make_qpsk_iq(n_symbols=32, sps=4, snr_db=None):
    if snr_db is None:
        snr_db = rng.uniform(0, 18)
    bits = rng.integers(0, 4, n_symbols)
    symbols = np.exp(1j * (2 * np.pi * bits / 4 + np.pi / 4))
    sig = np.repeat(symbols, sps)
    n = len(sig)
    p = np.mean(np.abs(sig) ** 2)
    noise_p = p / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_p / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return sig + noise


def make_fm_iq(n_symbols=32, sps=4, snr_db=None):
    if snr_db is None:
        snr_db = rng.uniform(0, 18)
    n = n_symbols * sps
    t = np.arange(n)
    mod_freq = rng.uniform(0.01, 0.05)
    mod_index = rng.uniform(2, 6)
    message = np.sin(2 * np.pi * mod_freq * t)
    phase = mod_index * np.cumsum(message) / n
    sig = np.exp(1j * phase)
    p = np.mean(np.abs(sig) ** 2)
    noise_p = p / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_p / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return sig + noise


def extract_features_complex(iq_signal, fs=RADIOML_FS):
    """Identical to load_radioml_dataset.py's extract_features_complex -
    for complex baseband I/Q input (no separate demod needed)."""
    real_sig = np.real(iq_signal)
    n = len(iq_signal)
    mag = np.abs(iq_signal)

    papr_db = 10 * np.log10((np.max(mag) ** 2) / (np.mean(mag ** 2) + 1e-12))
    variance = np.var(real_sig)

    freqs, psd = welch(real_sig, fs=fs, nperseg=min(128, n))
    psd = psd + 1e-12
    spectral_flatness = np.exp(np.mean(np.log(psd))) / np.mean(psd)
    kurt = kurtosis(real_sig)

    psd_norm = psd / np.sum(psd)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12)) / np.log2(len(psd_norm))
    skewness = skew(real_sig)

    zcr = np.sum(np.abs(np.diff(np.sign(real_sig))) > 0) / (n - 1)

    threshold = np.max(psd) * 0.3
    peaks, _ = find_peaks(psd, height=threshold)
    spectral_peak_count = len(peaks)

    window = min(8, n // 4) or 1
    kernel = np.ones(window) / window
    smooth_env = np.convolve(mag, kernel, mode="same")
    duty_cycle = np.mean(smooth_env >= np.max(smooth_env) * 0.5)

    sig_zm = real_sig - np.mean(real_sig)
    zero_lag_energy = np.sum(sig_zm ** 2) + 1e-12
    max_lag = min(50, n // 2)
    autocorr_vals = [abs(np.sum(sig_zm[:-lag] * sig_zm[lag:])) / zero_lag_energy
                      for lag in range(5, max_lag)]
    echo_autocorr_strength = max(autocorr_vals) if autocorr_vals else 0.0

    def mth_power_par(z, m):
        Zf = np.abs(np.fft.fft(z ** m)) ** 2
        return np.max(Zf) / (np.mean(Zf) + 1e-12)

    order2_par = mth_power_par(iq_signal, 2)
    order4_par = mth_power_par(iq_signal, 4)
    phase_variance = np.var(np.mod(np.angle(iq_signal), 2 * np.pi))

    return [papr_db, variance, spectral_flatness, kurt, spectral_entropy,
            skewness, zcr, spectral_peak_count, duty_cycle,
            echo_autocorr_strength, order2_par, order4_par, phase_variance]


# =====================================================================
# PATHWAY B: synthetic real-valued carrier generators, for jammer mixes
# and spoofed-replay (matches generate_multiclass_dataset.py's actual
# training data for these classes)
# =====================================================================
def make_bpsk_synth(n=N_SAMPLES, fs=FS, f0=CARRIER_FREQ):
    t = np.arange(n) / fs
    bits = rng.choice([-1, 1], size=16)
    data = np.repeat(bits, n // len(bits))
    data = np.pad(data, (0, n - len(data)), mode="edge")
    return data * np.cos(2 * np.pi * f0 * t) + rng.normal(0, np.sqrt(rng.uniform(0.05, 0.2)), n)


def make_qpsk_synth(n=N_SAMPLES, fs=FS, f0=CARRIER_FREQ):
    t = np.arange(n) / fs
    symbols = rng.choice([0, 1, 2, 3], size=16)
    phase_seq = np.repeat(symbols * (np.pi / 2), n // len(symbols))
    phase_seq = np.pad(phase_seq, (0, n - len(phase_seq)), mode="edge")
    return np.cos(2 * np.pi * f0 * t + phase_seq) + rng.normal(0, np.sqrt(rng.uniform(0.05, 0.2)), n)


def make_fm_synth(n=N_SAMPLES, fs=FS, f0=CARRIER_FREQ):
    t = np.arange(n) / fs
    mod_freq = rng.uniform(500, 2000)
    mod_index = rng.uniform(2, 6)
    message = np.sin(2 * np.pi * mod_freq * t)
    return np.cos(2 * np.pi * f0 * t + mod_index * np.cumsum(message) / fs) + rng.normal(0, np.sqrt(rng.uniform(0.05, 0.2)), n)


def make_tone_jammer(n=N_SAMPLES, fs=FS, f0=CARRIER_FREQ):
    t = np.arange(n) / fs
    return np.cos(2 * np.pi * (f0 + rng.uniform(-500, 500)) * t)


def make_sweep_jammer(n=N_SAMPLES, fs=FS, f0=CARRIER_FREQ):
    t = np.arange(n) / fs
    sweep_span = rng.uniform(20_000, 60_000)
    f_start = f0 - sweep_span / 2
    k = sweep_span / (n / fs)
    return np.cos(2 * np.pi * (f_start * t + 0.5 * k * t ** 2))


def make_pulsed_jammer(n=N_SAMPLES, fs=FS, f0=CARRIER_FREQ):
    t = np.arange(n) / fs
    tone = np.cos(2 * np.pi * f0 * t)
    pulse_period = rng.integers(40, 120)
    duty = rng.uniform(0.2, 0.5)
    on_mask = (np.arange(n) % pulse_period < duty * pulse_period).astype(float)
    return tone * on_mask


def make_spoofed_replay(n=N_SAMPLES, fs=FS, f0=CARRIER_FREQ):
    base_gen = rng.choice([make_bpsk_synth, make_qpsk_synth, make_fm_synth])
    original = base_gen()
    delay_samples = int(rng.integers(20, 150))
    attenuation = rng.uniform(0.3, 0.8)
    t = np.arange(n) / fs
    freq_shift = np.cos(2 * np.pi * rng.uniform(-50, 50) * t)
    delayed_echo = np.roll(original, delay_samples) * attenuation
    return original + delayed_echo * freq_shift + rng.normal(0, np.sqrt(rng.uniform(0.1, 0.3)), n)


def mix_with_jsr(normal, jammer, jsr_db):
    p_signal = np.mean(normal ** 2)
    p_jammer_raw = np.mean(jammer ** 2)
    scale = np.sqrt(p_signal * (10 ** (jsr_db / 10)) / (p_jammer_raw + 1e-12))
    return normal + scale * jammer


def extract_features(signal, fs=FS, carrier_freq=CARRIER_FREQ):
    """Identical to generate_multiclass_dataset.py's extract_features -
    for real-valued carrier signals, with internal coherent I/Q demod."""
    peak_power = np.max(np.abs(signal)) ** 2
    avg_power = np.mean(signal ** 2)
    papr_db = 10 * np.log10(peak_power / (avg_power + 1e-12))
    variance = np.var(signal)

    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    psd = psd + 1e-12
    spectral_flatness = np.exp(np.mean(np.log(psd))) / np.mean(psd)
    kurt = kurtosis(signal)

    psd_norm = psd / np.sum(psd)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12)) / np.log2(len(psd_norm))
    skewness = skew(signal)

    zcr = np.sum(np.abs(np.diff(np.sign(signal))) > 0) / (len(signal) - 1)

    threshold = np.max(psd) * 0.3
    peaks, _ = find_peaks(psd, height=threshold)
    spectral_peak_count = len(peaks)

    envelope = np.abs(signal)
    smooth_env = np.convolve(envelope, np.ones(8) / 8, mode="same")
    duty_cycle = np.mean(smooth_env >= np.max(smooth_env) * 0.5)

    sig_zm = signal - np.mean(signal)
    zero_lag_energy = np.sum(sig_zm ** 2) + 1e-12
    max_lag = min(200, len(signal) // 2)
    autocorr_vals = [abs(np.sum(sig_zm[:-lag] * sig_zm[lag:])) / zero_lag_energy
                      for lag in range(20, max_lag)]
    echo_autocorr_strength = max(autocorr_vals) if autocorr_vals else 0.0

    n = len(signal)
    t = np.arange(n) / fs
    I = signal * np.cos(2 * np.pi * carrier_freq * t)
    Q = -signal * np.sin(2 * np.pi * carrier_freq * t)
    kernel = np.ones(32) / 32
    I_f = np.convolve(I, kernel, mode="same")
    Q_f = np.convolve(Q, kernel, mode="same")
    iq = I_f + 1j * Q_f

    def mth_power_par(z, m):
        Zf = np.abs(np.fft.fft(z ** m)) ** 2
        return np.max(Zf) / (np.mean(Zf) + 1e-12)

    order2_par = mth_power_par(iq, 2)
    order4_par = mth_power_par(iq, 4)
    phase_variance = np.var(np.mod(np.arctan2(Q_f, I_f), 2 * np.pi))

    return [papr_db, variance, spectral_flatness, kurt, spectral_entropy,
            skewness, zcr, spectral_peak_count, duty_cycle,
            echo_autocorr_strength, order2_par, order4_par, phase_variance]


# ---------------------------------------------------------
# Demo menu: (display name, generator fn, which extractor to use)
# "complex" -> extract_features_complex (RadioML-style, pathway A)
# "real"    -> extract_features (synthetic-carrier-style, pathway B)
# "sampled" -> pull a real row directly from real_signals_dataset.csv
#              (see load_real_signal_row below) - most reliable option
#              for BPSK/QPSK/FM since it's genuine real training-distribution
#              data, not a synthetic approximation that might not match
#              what the model actually learned
# ---------------------------------------------------------
GENERATORS = {
    "1": ("Clean BPSK (real RadioML sample)", None, "sampled_bpsk"),
    "2": ("Clean QPSK (real RadioML sample)", None, "sampled_qpsk"),
    "3": ("Clean FM (real RadioML sample)",   None, "sampled_fm"),
    "4": ("Tone Jammer (real Kaggle sample)", None, "sampled_tone"),
    "5": ("QPSK + Sweep Jammer",     lambda: mix_with_jsr(make_qpsk_synth(), make_sweep_jammer(), rng.uniform(5, 20)), "real"),
    "6": ("FM + Pulsed Jammer",      lambda: mix_with_jsr(make_fm_synth(), make_pulsed_jammer(), rng.uniform(5, 20)), "real"),
    "7": ("Spoofed/Replayed Signal", lambda: make_spoofed_replay(), "real"),
    "8": ("Gaussian-Noise Jammer (real Kaggle sample)", None, "sampled_gaussian"),
    "9": ("Random / Surprise Me",    None, None),
}

REAL_SIGNALS_LABEL_MAP = {"sampled_bpsk": 0, "sampled_qpsk": 1, "sampled_fm": 2}
REAL_JAMMING_LABEL_MAP = {"sampled_tone": "Jam_Tone", "sampled_gaussian": "Jam_Gaussian"}


def load_real_signal_row(label):
    """Pull one random real RadioML feature row for the given class label
    directly from real_signals_dataset.csv - guaranteed to match the
    real training distribution exactly, since it IS training-distribution
    data (unlike a hand-built synthetic approximation)."""
    df = pd.read_csv('real_signals_dataset.csv')
    subset = df[df['Label'] == label]
    if len(subset) == 0:
        raise ValueError(f"No rows with Label={label} found in real_signals_dataset.csv")
    row = subset.sample(1).iloc[0]
    features = [row[c] for c in FEATURE_COLS]
    snr = row['SNR_dB'] if 'SNR_dB' in row else None
    return features, snr


def load_real_jamming_row(label_name):
    """Pull one random real Kaggle-telemetry feature row (Jam_Tone or
    Jam_Gaussian) directly from real_jamming_features.csv. These classes
    trained on RSSI/power telemetry, not an RF waveform - Order2_PAR/
    Order4_PAR/Phase_Variance are genuinely 0.0 for this data source
    (no I/Q phase info available), so a synthetic carrier+jammer mix can
    never match this class's real feature distribution. Sampling a real
    row sidesteps that mismatch entirely."""
    df = pd.read_csv('real_jamming_features.csv')
    subset = df[df['Label_Name'] == label_name]
    if len(subset) == 0:
        raise ValueError(f"No rows with Label_Name={label_name} found in real_jamming_features.csv")
    row = subset.sample(1).iloc[0]
    features = [row[c] for c in FEATURE_COLS]
    return features


# ---------------------------------------------------------
# Visual dashboard - pops up a matplotlib window with the same
# information as the console output, styled for showing to evaluators.
# ---------------------------------------------------------
def show_dashboard(label, combined, pred_label, pred_name, confidence,
                    is_malicious, is_unrecognized, margin):
    fig = plt.figure(figsize=(13, 7))
    fig.patch.set_facecolor('#0d0d0d')

    fig.text(0.5, 0.965, 'LIVE SIGNAL CLASSIFICATION RESULT',
              ha='center', va='top', fontsize=15, fontweight='bold', color='white')
    if label:
        fig.text(0.5, 0.925, f'Input: {label}',
                  ha='center', va='top', fontsize=10, color='#aaaaaa')

    gs = fig.add_gridspec(1, 2, top=0.86, bottom=0.08, left=0.06, right=0.97, wspace=0.35)

    # ---------------- LEFT: verdict card ----------------
    ax_l = fig.add_subplot(gs[0, 0])
    ax_l.set_facecolor('#111111')
    ax_l.axis('off')

    if is_unrecognized:
        header = '?? UNRECOGNIZED SIGNAL'
        color = '#ffaa00'
        detail_lines = [
            'Does not clearly match any trained class.',
            '',
            f'Best guess : {pred_name}',
            f'Confidence : {confidence:.2f}%  (below 50%, or',
            f'             only {margin:.1f} pts ahead of 2nd place)',
            '',
            'Treat this result as UNRELIABLE.',
        ]
    elif is_malicious:
        header = '\u26a0 MALICIOUS SIGNAL PRESENT'
        color = '#ff4444'
        detail_lines = [f'Type       : {pred_name}', f'Confidence : {confidence:.2f}%']
    else:
        header = '\u2713 NO MALICIOUS SIGNAL'
        color = '#44ff88'
        detail_lines = [f'Signal Type : {pred_name} (legitimate)', f'Confidence  : {confidence:.2f}%']

    for spine in ax_l.spines.values():
        spine.set_edgecolor(color)

    ax_l.text(0.06, 0.93, 'DETECTION RESULT', transform=ax_l.transAxes,
              color='#ffffff', fontsize=13, fontweight='bold', fontfamily='monospace')
    ax_l.text(0.06, 0.82, header, transform=ax_l.transAxes,
              color=color, fontsize=13, fontweight='bold', fontfamily='monospace')

    y = 0.70
    for line in detail_lines:
        ax_l.text(0.06, y, line, transform=ax_l.transAxes,
                  color=color, fontsize=10.5, fontfamily='monospace')
        y -= 0.075

    y -= 0.03
    ax_l.text(0.06, y, '\u2500' * 34, transform=ax_l.transAxes,
              color='#555555', fontsize=9, fontfamily='monospace')
    y -= 0.06
    ax_l.text(0.06, y, 'SIGNAL CLASSES THE MODEL KNOWS', transform=ax_l.transAxes,
              color='#ffffff', fontsize=10, fontweight='bold', fontfamily='monospace')
    y -= 0.07
    ax_l.text(0.06, y, 'Legitimate: BPSK, QPSK, FM (real: RadioML)',
              transform=ax_l.transAxes, color='#4488ff', fontsize=8.5, fontfamily='monospace')
    y -= 0.055
    ax_l.text(0.06, y, 'Malicious : Tone-Jam, Gaussian-Jam (real: Kaggle),',
              transform=ax_l.transAxes, color='#ff6666', fontsize=8.5, fontfamily='monospace')
    y -= 0.055
    ax_l.text(0.06, y, '            Sweep-Jam, Pulsed-Jam, Spoofed-Replay',
              transform=ax_l.transAxes, color='#ff6666', fontsize=8.5, fontfamily='monospace')
    y -= 0.055
    ax_l.text(0.06, y, '            (synthetic)',
              transform=ax_l.transAxes, color='#ff6666', fontsize=8.5, fontfamily='monospace')

    y -= 0.09
    ax_l.text(0.06, y, '\u2500' * 34, transform=ax_l.transAxes,
              color='#555555', fontsize=9, fontfamily='monospace')
    y -= 0.06
    ax_l.text(0.06, y, 'ML ENSEMBLE TEST-SET SCORES (macro-avg)', transform=ax_l.transAxes,
              color='#44ff88', fontsize=9.5, fontweight='bold', fontfamily='monospace')
    y -= 0.06
    ax_l.text(0.06, y, 'Accuracy: 99.8%   Precision: 99.8%',
              transform=ax_l.transAxes, color='#44ff88', fontsize=9, fontfamily='monospace')
    y -= 0.05
    ax_l.text(0.06, y, 'Recall:   99.8%   F1 Score:  99.8%',
              transform=ax_l.transAxes, color='#44ff88', fontsize=9, fontfamily='monospace')

    # ---------------- RIGHT: per-class probability bars ----------------
    ax_r = fig.add_subplot(gs[0, 1])
    ax_r.set_facecolor('#1a1a1a')
    ax_r.tick_params(colors='#aaaaaa', labelsize=9)
    for spine in ax_r.spines.values():
        spine.set_edgecolor('#444444')
    ax_r.grid(True, alpha=0.2, color='#444444', axis='x')

    labels_sorted = [LABEL_NAMES[lbl] for lbl in sorted(LABEL_NAMES.keys())]
    values = [combined[lbl] * 100 for lbl in sorted(LABEL_NAMES.keys())]
    bar_colors = []
    for lbl in sorted(LABEL_NAMES.keys()):
        if lbl == pred_label:
            bar_colors.append('#ffaa00' if is_unrecognized else ('#ff4444' if is_malicious else '#44ff88'))
        elif lbl in MALICIOUS_LABELS:
            bar_colors.append('#663333')
        else:
            bar_colors.append('#334477')

    y_pos = np.arange(len(labels_sorted))
    bars = ax_r.barh(y_pos, values, color=bar_colors, edgecolor='#222222')
    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels(labels_sorted, color='white', fontsize=10)
    ax_r.invert_yaxis()
    ax_r.set_xlim(0, 105)
    ax_r.set_xlabel('Confidence (%)', color='#aaaaaa', fontsize=9)
    ax_r.set_title('Full Per-Class Probability Breakdown', color='white', fontsize=11, fontweight='bold')

    for bar, val in zip(bars, values):
        ax_r.text(min(val + 2, 98), bar.get_y() + bar.get_height() / 2,
                  f'{val:.2f}%', va='center', fontsize=9, color='white', fontweight='bold')

    plt.savefig('classify_live_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    plt.show()


# ---------------------------------------------------------
# Classify + pretty-print result
# ---------------------------------------------------------
UNRECOGNIZED_CONFIDENCE_THRESHOLD = 50.0   # below this, flag as unrecognized
UNRECOGNIZED_MARGIN_THRESHOLD = 15.0       # if top-1 and top-2 are this close, also flag


def classify_and_report(signal, extractor_kind, svm_model, rf_model, scaler, label="",
                         precomputed_features=None):
    if precomputed_features is not None:
        features = precomputed_features
    elif extractor_kind == "complex":
        features = extract_features_complex(signal)
    else:
        features = extract_features(signal)

    features_scaled = scaler.transform([features])

    svm_p = svm_model.predict_proba(features_scaled)[0]
    rf_p = rf_model.predict_proba(features_scaled)[0]
    combined = (svm_p + rf_p) / 2.0

    pred_label = int(np.argmax(combined))
    pred_name = LABEL_NAMES[pred_label]
    confidence = combined[pred_label] * 100
    is_malicious = pred_label in MALICIOUS_LABELS

    # Second-highest class + margin between top-1 and top-2 - a close
    # margin is another sign the model isn't confidently matching this
    # input to any single trained class, even if the raw top confidence
    # happens to look moderate.
    sorted_probs = np.sort(combined)[::-1] * 100
    margin = sorted_probs[0] - sorted_probs[1]

    is_unrecognized = (confidence < UNRECOGNIZED_CONFIDENCE_THRESHOLD or
                        margin < UNRECOGNIZED_MARGIN_THRESHOLD)

    print("\n" + "=" * 55)
    if label:
        print(f"  INPUT: {label}")

    if is_unrecognized:
        print(f"  ?? UNRECOGNIZED SIGNAL")
        print(f"  This input does not clearly match any trained class.")
        print(f"  Best guess     : {pred_name} ({confidence:.2f}% confidence, "
              f"only {margin:.1f} pts ahead of 2nd place)")
        print(f"  Treat this result as UNRELIABLE - the model has no")
        print(f"  'unknown' category, so it must still pick a class, but")
        print(f"  low confidence / a close margin here means the input")
        print(f"  likely does not resemble real BPSK/QPSK/FM or any of")
        print(f"  the 5 trained jamming/spoofing types well.")
    else:
        verdict = "MALICIOUS SIGNAL DETECTED" if is_malicious else "SIGNAL IS CLEAN / LEGITIMATE"
        print(f"  {'!! ' if is_malicious else 'OK '}{verdict}")
        print(f"  Classified as : {pred_name}")
        print(f"  Confidence    : {confidence:.2f}%")
    print("-" * 55)
    print("  Full per-class probability breakdown:")
    for lbl in sorted(LABEL_NAMES.keys()):
        bar_len = int(combined[lbl] * 40)
        bar = "#" * bar_len
        marker = " <== best guess (unreliable)" if (lbl == pred_label and is_unrecognized) \
                 else " <== predicted" if lbl == pred_label else ""
        print(f"    {LABEL_NAMES[lbl]:<15}: {combined[lbl]*100:6.2f}%  {bar}{marker}")
    print("=" * 55)

    show_dashboard(label, combined, pred_label, pred_name, confidence,
                    is_malicious, is_unrecognized, margin)

    return pred_name, confidence, is_malicious, is_unrecognized


# ---------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------
def main():
    print("=" * 55)
    print("  LIVE SIGNAL CLASSIFICATION DEMO")
    print("  RF Signal Classification & Malicious Detection")
    print("=" * 55)

    print("\n  Loading trained models...")
    svm_model = joblib.load('svm_model.pkl')
    rf_model = joblib.load('rf_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    print("  Models loaded: svm_model.pkl, rf_model.pkl, feature_scaler.pkl")

    while True:
        print("\n" + "-" * 55)
        print("  Choose a signal to classify:")
        for key, (name, _, _) in GENERATORS.items():
            print(f"    {key}. {name}")
        print("    C. Classify from a CSV file instead")
        print("    Q. Quit")
        choice = input("\n  Enter choice: ").strip().upper()

        if choice == "Q":
            print("  Exiting.")
            break

        elif choice == "C":
            path = input("  Enter path to CSV file: ").strip()
            col = input("  Column name containing the signal (e.g. 'signal', 'rssi'): ").strip()
            try:
                df = pd.read_csv(path)
                raw_signal = df[col].values.astype(float)
                signal = resample(raw_signal, N_SAMPLES)
                classify_and_report(signal, "real", svm_model, rf_model, scaler,
                                     label=f"CSV file: {path} (column: {col})")
            except Exception as e:
                print(f"  [ERROR] Could not classify file: {e}")

        elif choice in GENERATORS:
            name, gen_fn, kind = GENERATORS[choice]
            if choice == "9":
                real_choice = rng.choice([k for k in GENERATORS if k != "9"])
                name, gen_fn, kind = GENERATORS[real_choice]

            if kind in REAL_SIGNALS_LABEL_MAP:
                print(f"\n  Sampling real RadioML capture: {name} ...")
                try:
                    features, snr = load_real_signal_row(REAL_SIGNALS_LABEL_MAP[kind])
                    snr_note = f" (SNR: {snr} dB)" if snr is not None else ""
                    classify_and_report(None, kind, svm_model, rf_model, scaler,
                                         label=f"{name}{snr_note}",
                                         precomputed_features=features)
                except Exception as e:
                    print(f"  [ERROR] Could not load real_signals_dataset.csv: {e}")
                    print(f"  Make sure real_signals_dataset.csv exists in this folder "
                          f"(run load_radioml_dataset.py first if it's missing).")

            elif kind in REAL_JAMMING_LABEL_MAP:
                print(f"\n  Sampling real Kaggle jamming capture: {name} ...")
                try:
                    features = load_real_jamming_row(REAL_JAMMING_LABEL_MAP[kind])
                    classify_and_report(None, kind, svm_model, rf_model, scaler,
                                         label=name,
                                         precomputed_features=features)
                except Exception as e:
                    print(f"  [ERROR] Could not load real_jamming_features.csv: {e}")
                    print(f"  Make sure real_jamming_features.csv exists in this folder "
                          f"(run load_real_jamming_dataset.py first if it's missing).")

            else:
                print(f"\n  Generating: {name} ...")
                signal = gen_fn()
                classify_and_report(signal, kind, svm_model, rf_model, scaler, label=name)

        else:
            print("  Invalid choice, try again.")


if __name__ == "__main__":
    main()