# =========================================
# classify_live.py  (v2)
# Interactive LIVE demo: generate a FRESH signal (never in any training set), classify it with
# the trained SVM + Random Forest ensemble and compare with the known ground truth.
#
# Model  : ensemble_bundle.pkl (single-source synthetic benchmark; run ml_classifier.py on
#          rf_synthetic8_dataset.csv first). Valid for SNR ~4-10 dB, JSR 5-20 dB.
# Needs  : generate_multiclass_dataset_v2.py, cfar_core.py, ensemble_bundle.pkl
#          (multiclass_results.csv is optional, for the score card)
#
# What changed vs v1
#  1. v1 used the hybrid pipeline and, for BPSK/QPSK/FM/Tone/Gaussian, pulled feature rows
#     straight out of the training CSVs ("sampled" options) -> the demo classified training data.
#     v2 generates every class fresh with the same generator and feature code as training, so the
#     result is a real (in-distribution) test, and it prints ground truth vs prediction.
#  2. Session tally (correct / wrong / unrecognised) so the audience sees honest accuracy.
#  3. Stress test (S): choose SNR / JSR yourself, e.g. outside the training range, and watch the
#     accuracy fall (see Table IV of the paper).
#  4. External capture (C) uses cfar_core.detect (identical to cacfar.py v2). No CFAR region ->
#     benign, classifier not invoked. Otherwise all windows of the capture are classified.
#     v1 had its own old CFAR copy and classified only the IFFT-isolated band.
#  5. Models from the bundle (scaler inside the pipelines, class order checked). Score card reads
#     multiclass_results.csv instead of hard-coded 99.8%. No hard-coded path.
#  6. The I/Q-file option was removed: the synthetic model expects real-valued passband windows
#     (1 MHz, 204.8 kHz carrier), not RadioML-style complex frames.
# =========================================

import os
import sys

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
DATA_DIR = os.environ.get("RF_DATA_DIR", _HERE)
os.chdir(DATA_DIR)
sys.path.insert(0, DATA_DIR)
sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.signal import resample

import generate_multiclass_dataset_v2 as G
import cfar_core

SHOW_PLOT = os.environ.get("RF_SHOW_PLOT", "1") == "1"
CAPTURE_FS = 2.048e6
# demo heuristics (not validated thresholds): flag a result as UNRECOGNISED if the best class has
# less than 50% probability or leads the runner-up by less than 15 points
CONF_TH, MARGIN_TH = 50.0, 15.0

MENU = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7}


# ---------------------------------------------------------------- signals
def make_signal(label, snr_db=None, jsr_db=None):
    """Fresh 1024-sample window at 1 MHz, same generator as the training data."""
    nv = None if snr_db is None else 0.5 / (10 ** (snr_db / 10))
    if label in G.LEGIT_GENERATORS:
        return G.LEGIT_GENERATORS[label](noise_var=nv)
    if label == 6:
        return G.make_spoofed_replay(noise_var=nv)[0]
    base = G.LEGIT_GENERATORS[int(G.rng.choice([0, 1, 2]))](noise_var=nv)
    jsr = G.rng.uniform(5, 20) if jsr_db is None else jsr_db
    return G.mix_with_jsr(base, G.JAM_GENERATORS[label](), jsr)


def predict(bundle, windows):
    """windows: list of 1024-sample arrays -> mean ensemble probability vector."""
    X = pd.DataFrame([G.extract_features(w) for w in windows], columns=bundle["feature_cols"])
    p = (bundle["svm_pipeline"].predict_proba(X.values) + bundle["rf_pipeline"].predict_proba(X.values)) / 2
    return p.mean(axis=0)


def decide(proba, label_names, malicious):
    order = np.argsort(proba)[::-1]
    best = int(order[0])
    p = proba[best] * 100
    margin = (proba[order[0]] - proba[order[1]]) * 100
    unrec = p < CONF_TH or margin < MARGIN_TH
    verdict = "UNRECOGNISED" if unrec else ("MALICIOUS" if best in malicious else "LEGITIMATE")
    return best, p, margin, verdict


# ---------------------------------------------------------------- output
def scores_text(results_csv="multiclass_results.csv"):
    try:
        r = pd.read_csv(results_csv)
        e = r[r["Model"] == "Ensemble"].iloc[0]
        return (f"Accuracy {e['Accuracy']*100:.2f}%  Precision {e['Precision_macro']*100:.2f}%",
                f"Recall   {e['Recall_macro']*100:.2f}%  F1        {e['F1_macro']*100:.2f}%")
    except Exception:
        return ("(multiclass_results.csv not found)", "")


def report(label, truth, proba, bundle):
    names = {int(k): v for k, v in bundle["label_names"].items()}
    mal = set(bundle["malicious_labels"])
    best, p, margin, verdict = decide(proba, names, mal)
    print("\n" + "=" * 58)
    print(f"  INPUT        : {label}")
    if truth is not None:
        print(f"  GROUND TRUTH : {names[truth]}")
    print(f"  PREDICTED    : {names[best]}  (model-estimated probability {p:.2f}%, "
          f"{margin:.1f} pts ahead of 2nd)")
    print(f"  VERDICT      : {verdict}")
    if verdict == "UNRECOGNISED":
        print("  -> low probability or close margin: treat as unreliable. The model has no")
        print("     'unknown' class, so it must still pick one of the 8 trained classes.")
    if truth is not None:
        print(f"  CHECK        : {'CORRECT' if best == truth else 'WRONG'}")
    print("-" * 58)
    for k in sorted(names):
        print(f"    {names[k]:<15}: {proba[k]*100:6.2f}%  {'#' * int(proba[k] * 40)}"
              f"{'  <== predicted' if k == best else ''}")
    print("=" * 58)
    dashboard(label, truth, proba, best, p, margin, verdict, names, mal)
    return best, verdict


def dashboard(label, truth, proba, best, p, margin, verdict, names, mal):
    fig = plt.figure(figsize=(13, 7))
    fig.patch.set_facecolor('#0d0d0d')
    fig.text(0.5, 0.965, 'LIVE SIGNAL CLASSIFICATION RESULT', ha='center', va='top',
             fontsize=15, fontweight='bold', color='white')
    fig.text(0.5, 0.925, f'Input: {label}', ha='center', va='top', fontsize=10, color='#aaaaaa')
    gs = fig.add_gridspec(1, 2, top=0.86, bottom=0.08, left=0.06, right=0.97, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor('#111111'); ax.axis('off')
    color = {'MALICIOUS': '#ff4444', 'LEGITIMATE': '#44ff88', 'UNRECOGNISED': '#ffaa00'}[verdict]
    head = {'MALICIOUS': 'MALICIOUS SIGNAL PRESENT', 'LEGITIMATE': 'NO MALICIOUS SIGNAL',
            'UNRECOGNISED': 'UNRECOGNISED - UNRELIABLE'}[verdict]
    lines = [('DETECTION RESULT', '#ffffff', 13, 'bold'), ('', 'white', 8, 'normal'),
             (head, color, 12, 'bold'),
             (f'Predicted : {names[best]}', color, 10.5, 'normal'),
             (f'Probability (model-estimated): {p:.1f}%', color, 10, 'normal'),
             (f'Lead over 2nd place: {margin:.1f} pts', color, 10, 'normal')]
    if truth is not None:
        ok = (best == truth)
        lines.append((f'Ground truth: {names[truth]}  ->  {"CORRECT" if ok else "WRONG"}',
                      '#44ff88' if ok else '#ff4444', 10.5, 'bold'))
    lines += [('', 'white', 6, 'normal'), ('-' * 40, '#555555', 8, 'normal'),
              ('TRAINED CLASSES', '#ffffff', 10, 'bold'),
              ('Legitimate: BPSK, QPSK, FM', '#4488ff', 9, 'normal'),
              ('Malicious : Tone, Sweep, Pulsed, Gaussian,', '#ff6666', 9, 'normal'),
              ('            Spoofed-Replay', '#ff6666', 9, 'normal'),
              ('(all synthetic; one generator)', '#aaaaaa', 8, 'normal'),
              ('', 'white', 6, 'normal'), ('-' * 40, '#555555', 8, 'normal'),
              ('ENSEMBLE TEST-SET SCORES (macro-avg)', '#44ff88', 9.5, 'bold')]
    for s in scores_text():
        lines.append((s, '#44ff88', 9, 'normal'))
    lines += [('', 'white', 6, 'normal'),
              ('Valid for SNR ~4-10 dB, JSR 5-20 dB.', '#ffcc00', 8, 'normal'),
              ('Outside this range accuracy drops (paper, Table IV).', '#ffcc00', 8, 'normal')]
    y = 0.95
    for text, c, size, w in lines:
        ax.text(0.06, y, text, transform=ax.transAxes, color=c, fontsize=size, fontweight=w,
                va='top', fontfamily='monospace')
        y -= 0.052 if size < 11 else 0.07
    for s in ax.spines.values():
        s.set_edgecolor(color)

    ar = fig.add_subplot(gs[0, 1]); ar.set_facecolor('#1a1a1a')
    ar.tick_params(colors='#aaaaaa', labelsize=9)
    for s in ar.spines.values():
        s.set_edgecolor('#444444')
    ar.grid(True, alpha=0.2, color='#444444', axis='x')
    ks = sorted(names)
    cols = [(('#ffaa00' if verdict == 'UNRECOGNISED' else '#ff4444' if k in mal else '#44ff88')
             if k == best else ('#663333' if k in mal else '#334477')) for k in ks]
    bars = ar.barh(np.arange(len(ks)), [proba[k] * 100 for k in ks], color=cols, edgecolor='#222222')
    ar.set_yticks(np.arange(len(ks))); ar.set_yticklabels([names[k] for k in ks], color='white', fontsize=10)
    ar.invert_yaxis(); ar.set_xlim(0, 105)
    ar.set_xlabel('Model-estimated probability (%)', color='#aaaaaa', fontsize=9)
    ar.set_title('Per-class probability', color='white', fontsize=11, fontweight='bold')
    for b, k in zip(bars, ks):
        ar.text(min(proba[k] * 100 + 2, 98), b.get_y() + b.get_height() / 2, f'{proba[k]*100:.2f}%',
                va='center', fontsize=9, color='white', fontweight='bold')
    plt.savefig('classify_live_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    if SHOW_PLOT:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------- external capture
def classify_capture(path, col, fs, bundle):
    x = pd.read_csv(path)[col].values.astype(float)
    res = cfar_core.detect(x, fs)
    if not res["detected"]:
        print("\n  [CFAR] No region above the CA-CFAR threshold -> capture declared BENIGN,")
        print("         classifier not invoked.")
        return None
    print(f"\n  [CFAR] centre {res['center_freq_hz']/1e3:.2f} kHz | bandwidth {res['bandwidth_hz']:.0f} Hz"
          f" (bin width {res['freq_res']/1e3:.1f} kHz) | peak {res['peak_db']:.2f} dB"
          f" | noise floor {res['noise_floor_db']:.2f} dB | {len(res['regions'])} region(s)")
    win_len = max(2, int(G.N_SAMPLES / G.FS * fs))
    n_win = max(1, len(x) // win_len)
    windows = [resample(x[i * win_len:(i + 1) * win_len], G.N_SAMPLES) for i in range(n_win)]
    proba = predict(bundle, windows)
    print(f"  [CLASSIFY] {n_win} window(s) of {win_len} samples resampled to {G.N_SAMPLES}; "
          "probabilities averaged. Ground truth unknown for external captures.")
    return report(f"capture: {path} [{col}]", None, proba, bundle)


# ---------------------------------------------------------------- main
def ask_float(prompt):
    s = input(prompt).strip()
    return float(s) if s else None


def main():
    print("=" * 58)
    print("  LIVE SIGNAL CLASSIFICATION DEMO (v2)")
    print("  Fresh synthetic signals - never part of the training data")
    print("=" * 58)
    bundle = joblib.load("ensemble_bundle.pkl")
    assert list(bundle["svm_pipeline"].classes_) == list(range(8))
    names = {int(k): v for k, v in bundle["label_names"].items()}
    tally = {"correct": 0, "wrong": 0, "unrecognised": 0}

    while True:
        print("\n" + "-" * 58)
        for k, c in MENU.items():
            print(f"    {k}. {names[c]}")
        print("    9. Random class")
        print("    S. Stress test (choose SNR / JSR)")
        print("    C. Classify a captured real-valued signal (CFAR first)")
        print("    Q. Quit")
        ch = input("\n  Enter choice: ").strip().upper()
        if ch == "Q":
            break
        if ch == "C":
            path = input("  CSV path: ").strip()
            col = input("  Waveform column [Mixed_Signal]: ").strip() or "Mixed_Signal"
            fs = ask_float(f"  Sample rate in Hz [{CAPTURE_FS:.0f}]: ") or CAPTURE_FS
            try:
                classify_capture(path, col, fs, bundle)
            except Exception as e:
                print(f"  [ERROR] {e}")
            continue

        snr = jsr = None
        if ch == "S":
            sel = input("  Class number 1-8: ").strip()
            if sel not in MENU:
                print("  Invalid class.")
                continue
            truth = MENU[sel]
            snr = ask_float("  SNR in dB (blank = training range): ")
            if truth in (3, 4, 5, 7):
                jsr = ask_float("  JSR in dB (blank = training range): ")
        elif ch == "9":
            truth = int(G.rng.integers(0, 8))
        elif ch in MENU:
            truth = MENU[ch]
        else:
            print("  Invalid choice.")
            continue

        sig = make_signal(truth, snr, jsr)
        note = (f" [SNR {snr} dB]" if snr is not None else "") + (f" [JSR {jsr} dB]" if jsr is not None else "")
        print(f"\n  Generated fresh {names[truth]}{note} (classifier stage only; CFAR is shown with option C)")
        proba = predict(bundle, [sig])
        best, verdict = report(f"{names[truth]} (fresh){note}", truth, proba, bundle)
        if verdict == "UNRECOGNISED":
            tally["unrecognised"] += 1
        elif best == truth:
            tally["correct"] += 1
        else:
            tally["wrong"] += 1
        n = sum(tally.values())
        print(f"  Session: {tally['correct']} correct, {tally['wrong']} wrong, "
              f"{tally['unrecognised']} unrecognised (of {n})")

    n = sum(tally.values())
    if n:
        print(f"\n  Final session tally: {tally['correct']}/{n} correct, {tally['wrong']} wrong, "
              f"{tally['unrecognised']} unrecognised.")
    print("  Exiting.")


if __name__ == "__main__":
    main()