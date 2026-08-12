"""
merge_final_hybrid_dataset.py

Combines THREE data sources into your final 7-class training set:

  1. real_signals_dataset.csv  - REAL BPSK/QPSK/FM from RadioML2016.10a
  2. real_jamming_features.csv - REAL Tone-Jam from active_scan/malicious/
                                  singletone/0dbm (Normal_Background rows
                                  from this file are NOT used here - they
                                  overlap conceptually with BPSK/QPSK/FM
                                  and would create a confusing extra class)
  3. rf_multiclass_dataset.csv - SYNTHETIC Sweep-Jam, Pulsed-Jam,
                                  Spoofed-Replay (no real dataset exists
                                  for these attack types)

Output: rf_final_hybrid_dataset.csv

Run:
    python merge_final_hybrid_dataset.py
"""

import pandas as pd

FEATURE_COLS = ["PAPR_dB", "Variance", "Spectral_Flatness", "Kurtosis",
                "Spectral_Entropy", "Skewness", "ZCR",
                "Spectral_Peak_Count", "Envelope_Duty_Cycle",
                "Echo_Autocorr_Strength", "Order2_PAR", "Order4_PAR", "Phase_Variance"]

SYNTHETIC_CLASSES_TO_KEEP = ["Jam_Sweep", "Jam_Pulsed", "Spoofed_Replay"]

# --- 1. Real signal types (BPSK/QPSK/FM) from RadioML ---
real_signals_df = pd.read_csv("real_signals_dataset.csv")
real_signals_df = real_signals_df[FEATURE_COLS + ["Label", "Label_Name"]].copy()
real_signals_df["Source"] = "RadioML2016.10a"

# --- 2. Real Tone-Jam + Gaussian-Jam from active scan dataset ---
real_jam_df = pd.read_csv("real_jamming_features.csv")
real_jam_classes = real_jam_df[real_jam_df["Label_Name"].isin(["Jam_Tone", "Jam_Gaussian"])].copy()
# real_jamming_features.csv doesn't have Order2_PAR/Order4_PAR/Phase_Variance
# as real values (set to 0 in load_real_jamming_dataset.py, since RSSI
# telemetry has no I/Q or carrier phase to compute them from) - keep as
# is, this is an honest, documented limitation, not an error
missing_cols = [c for c in FEATURE_COLS if c not in real_jam_classes.columns]
for c in missing_cols:
    real_jam_classes[c] = 0.0
real_jam_classes = real_jam_classes[FEATURE_COLS + ["Label", "Label_Name"]].copy()
real_jam_classes["Source"] = "Active_Scan_Kaggle"

# --- 3. Synthetic Sweep/Pulsed/Spoofed (no real equivalent exists) ---
synthetic_df = pd.read_csv("rf_multiclass_dataset.csv")
synthetic_df = synthetic_df[synthetic_df["Label_Name"].isin(SYNTHETIC_CLASSES_TO_KEEP)].copy()
synthetic_df = synthetic_df[FEATURE_COLS + ["Label", "Label_Name"]].copy()
synthetic_df["Source"] = "Synthetic"

# --- Combine ---
hybrid_df = pd.concat([real_signals_df, real_jam_classes, synthetic_df], ignore_index=True)
hybrid_df = hybrid_df.sample(frac=1, random_state=42).reset_index(drop=True)

hybrid_df.to_csv("rf_final_hybrid_dataset.csv", index=False)

print(f"Saved {len(hybrid_df)} rows to rf_final_hybrid_dataset.csv")
print(f"\nClass breakdown:")
print(hybrid_df["Label_Name"].value_counts())
print(f"\nSource breakdown:")
print(hybrid_df["Source"].value_counts())
print(f"\n--- IMPORTANT: class balance check ---")
counts = hybrid_df["Label_Name"].value_counts()
print(f"Largest class: {counts.idxmax()} ({counts.max()} rows)")
print(f"Smallest class: {counts.idxmin()} ({counts.min()} rows)")
if counts.max() / counts.min() > 3:
    print("WARNING: significant class imbalance (>3x). Consider using")
    print("class_weight='balanced' in SVC/RandomForestClassifier, or")
    print("downsampling larger classes, before training on this data.")