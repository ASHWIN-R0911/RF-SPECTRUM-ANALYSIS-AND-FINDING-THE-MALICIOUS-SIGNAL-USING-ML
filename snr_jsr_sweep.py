"""
SNR / JSR robustness sweep
Needs (same folder): generate_multiclass_dataset_v2.py, ensemble_bundle.pkl
                     (run ml_classifier.py on rf_synthetic8_dataset.csv first)

Fresh signals are generated for every sweep point (never seen in training), the
saved ensemble classifies them, and accuracy is reported per condition.

  Sweep A - SNR : legitimate-signal SNR varied, jammer JSR fixed (JSR_FIXED_DB)
  Sweep B - JSR : jammer-to-signal ratio varied, SNR fixed (SNR_FIXED_DB)

SNR definition: signal power is ~0.5 (unit-amplitude carrier), so
    noise_var = 0.5 / 10^(SNR/10).
The training data covers SNR ~4-10 dB and JSR 5-20 dB; points outside that
range are OUT-OF-DISTRIBUTION tests. Retrain on a wider range if you want the
curve to hold there, and say which you did in the paper.

Output: snr_jsr_sweep.csv, snr_jsr_sweep.png
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
DATA_DIR = os.environ.get("RF_DATA_DIR", _HERE)
os.chdir(DATA_DIR)
import sys
sys.path.insert(0, DATA_DIR)
import generate_multiclass_dataset_v2 as G

N_PER_CLASS = 150
SNR_GRID_DB = [-10, -5, 0, 5, 10, 15, 20]
JSR_GRID_DB = [-10, -5, 0, 5, 10, 15, 20]
JSR_FIXED_DB = 12.0
SNR_FIXED_DB = 7.0
SHOW_PLOT = True

bundle = joblib.load("ensemble_bundle.pkl")
cols = bundle["feature_cols"]
names = bundle["label_names"]
classes = bundle["class_order"]
svm_p, rf_p = bundle["svm_pipeline"], bundle["rf_pipeline"]


def noise_var_from_snr(snr_db):
    return 0.5 / (10 ** (snr_db / 10))


def make_sample(label, snr_db, jsr_db):
    nv = noise_var_from_snr(snr_db)
    if label in G.LEGIT_GENERATORS:
        return G.LEGIT_GENERATORS[label](noise_var=nv)
    if label == 6:
        return G.make_spoofed_replay(noise_var=nv)[0]
    base = G.LEGIT_GENERATORS[int(G.rng.choice([0, 1, 2]))](noise_var=nv)
    return G.mix_with_jsr(base, G.JAM_GENERATORS[label](), jsr_db)


def evaluate(label_list, snr_db, jsr_db):
    X, y = [], []
    for label in label_list:
        for _ in range(N_PER_CLASS):
            X.append(G.extract_features(make_sample(label, snr_db, jsr_db)))
            y.append(label)
    X = pd.DataFrame(X, columns=cols)
    proba = (svm_p.predict_proba(X.values) + rf_p.predict_proba(X.values)) / 2
    pred = np.array(classes)[proba.argmax(axis=1)]
    y = np.array(y)
    per_class = {names[c]: float((pred[y == c] == c).mean()) for c in label_list}
    return float((pred == y).mean()), per_class


rows = []
print("Sweep A - SNR (JSR fixed at %.0f dB)" % JSR_FIXED_DB)
for snr in SNR_GRID_DB:
    acc, pc = evaluate(classes, snr, JSR_FIXED_DB)
    rows.append({"Sweep": "SNR", "Value_dB": snr, "Accuracy": acc, **pc})
    print(f"  SNR {snr:>4} dB : accuracy {acc*100:6.2f}%")

print("Sweep B - JSR (SNR fixed at %.0f dB), jammer classes only" % SNR_FIXED_DB)
JAMMERS = [3, 4, 5, 7]
for jsr in JSR_GRID_DB:
    acc, pc = evaluate(JAMMERS, SNR_FIXED_DB, jsr)
    rows.append({"Sweep": "JSR", "Value_dB": jsr, "Accuracy": acc, **pc})
    print(f"  JSR {jsr:>4} dB : jammer-class accuracy {acc*100:6.2f}%")

res = pd.DataFrame(rows)
res.to_csv("snr_jsr_sweep.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for ax, sweep, xlabel, ttl in [
        (axes[0], "SNR", "SNR (dB)", f"Accuracy vs SNR (JSR fixed {JSR_FIXED_DB:.0f} dB)"),
        (axes[1], "JSR", "JSR (dB)", f"Jammer-class accuracy vs JSR (SNR fixed {SNR_FIXED_DB:.0f} dB)")]:
    d = res[res.Sweep == sweep]
    ax.plot(d.Value_dB, d.Accuracy * 100, "k-o", lw=2.5, label="Overall")
    for c in classes:
        n = names[c]
        if n in d.columns and d[n].notna().any():
            ax.plot(d.Value_dB, d[n] * 100, "--", lw=1, marker=".", label=n)
    ax.set_title(ttl)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim([0, 102])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
snr_r = bundle.get("snr_range") or (4, 10)      # narrow dataset: SNR ~4-10 dB (not recorded)
jsr_r = bundle.get("jsr_range") or (5, 20)
axes[0].axvspan(snr_r[0], snr_r[1], color="green", alpha=0.08, label="training SNR range")
axes[1].axvspan(jsr_r[0], jsr_r[1], color="green", alpha=0.08)
plt.tight_layout()
plt.savefig("snr_jsr_sweep.png", dpi=150, bbox_inches="tight")
if SHOW_PLOT:
    plt.show()
print("Saved snr_jsr_sweep.csv, snr_jsr_sweep.png (green band = training range)")