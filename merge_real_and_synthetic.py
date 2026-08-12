"""
merge_real_and_synthetic.py

Combines real_signals_dataset.csv (real BPSK/QPSK/FM from RadioML2016.10a)
with the synthetic jammer/spoofing rows from rf_multiclass_dataset.csv
(Tone-Jam, Sweep-Jam, Pulsed-Jam, Spoofed-Replay - these stay synthetic
since no real-world dataset exists for your specific jamming/replay setup).

Output: rf_hybrid_dataset.csv - point multiclass_classifier.py at this
file instead of rf_multiclass_dataset.csv to train on the hybrid data.

Run:
    python merge_real_and_synthetic.py
"""

import pandas as pd

FEATURE_COLS = ["PAPR_dB", "Variance", "Spectral_Flatness", "Kurtosis",
                "Spectral_Entropy", "Skewness", "ZCR",
                "Spectral_Peak_Count", "Envelope_Duty_Cycle",
                "Echo_Autocorr_Strength", "Phase_State_Count", "Phase_Variance"]

MALICIOUS_CLASSES = ["Jam_Tone", "Jam_Sweep", "Jam_Pulsed", "Spoofed_Replay"]

# --- Load real signal data ---
real_df = pd.read_csv("real_signals_dataset.csv")
real_df = real_df[FEATURE_COLS + ["Label", "Label_Name"]].copy()
real_df["Source"] = "real"

# --- Load synthetic dataset, keep ONLY the malicious rows (jammers + spoofing) ---
synthetic_df = pd.read_csv("rf_multiclass_dataset.csv")
malicious_df = synthetic_df[synthetic_df["Label_Name"].isin(MALICIOUS_CLASSES)].copy()
malicious_df = malicious_df[FEATURE_COLS + ["Label", "Label_Name"]].copy()
malicious_df["Source"] = "synthetic"

# --- Combine ---
hybrid_df = pd.concat([real_df, malicious_df], ignore_index=True)
hybrid_df = hybrid_df.sample(frac=1, random_state=42).reset_index(drop=True)

hybrid_df.to_csv("rf_hybrid_dataset.csv", index=False)

print(f"Saved {len(hybrid_df)} rows to rf_hybrid_dataset.csv")
print(f"\nClass breakdown:")
print(hybrid_df["Label_Name"].value_counts())
print(f"\nSource breakdown:")
print(hybrid_df["Source"].value_counts())
