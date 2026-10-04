# =========================================
# Quick verification: does the Order2_PAR / Order4_PAR fix actually
# improve BPSK vs QPSK separation on your REAL RadioML data?
# Run this AFTER load_radioml_dataset.py (needs real_signals_dataset.csv).
# This does NOT need the merged hybrid dataset - it's a fast sanity check
# on just the 3 real classes (BPSK, QPSK, FM) before you update the other
# two feature-extraction scripts and re-run the full pipeline.
# =========================================

import os
os.chdir(r'E:\D\projectsml')   # set your path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('real_signals_dataset.csv')

FEATURE_COLS = ['PAPR_dB', 'Variance', 'Spectral_Flatness', 'Kurtosis',
                'Spectral_Entropy', 'Skewness', 'ZCR',
                'Spectral_Peak_Count', 'Envelope_Duty_Cycle',
                'Echo_Autocorr_Strength', 'Order2_PAR', 'Order4_PAR', 'Phase_Variance']

X = df[FEATURE_COLS].values
y = df['Label'].values
LABEL_NAMES = {0: 'BPSK', 1: 'QPSK', 2: 'FM'}

print(f"Loaded {len(df)} real samples: "
      f"{sum(y==0)} BPSK, {sum(y==1)} QPSK, {sum(y==2)} FM")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=150, max_depth=12,
                              random_state=42, class_weight='balanced', n_jobs=-1)
clf.fit(X_train_s, y_train)
pred = clf.predict(X_test_s)

print(f"\nOverall accuracy: {accuracy_score(y_test, pred)*100:.2f}%")
print(f"\nClassification report:")
print(classification_report(y_test, pred, target_names=['BPSK', 'QPSK', 'FM']))

cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
print("Confusion matrix (rows=true, cols=predicted):")
print("            BPSK  QPSK  FM")
for i, name in enumerate(['BPSK', 'QPSK', 'FM']):
    print(f"  {name:<8}: {cm[i]}")

bpsk_recall = cm[0, 0] / cm[0].sum()
qpsk_recall = cm[1, 1] / cm[1].sum()
print(f"\nBPSK recall: {bpsk_recall*100:.2f}%")
print(f"QPSK recall: {qpsk_recall*100:.2f}%")
print(f"(Compare this to your old ~90% each with Phase_State_Count)")

cv = cross_val_score(clf, X_train_s, y_train, cv=5)
print(f"\n5-fold CV accuracy: {cv.mean()*100:.2f}% (+/- {cv.std()*100:.2f}%)")
