# =========================================
# Multi-Class RF Signal Classifier
# Project: RF Signal Classification and Malicious Signal Detection
# Input  : rf_multiclass_dataset.csv
# Output : CONFUSION MATRIX ONLY (all other outputs commented out)
# Models : SVM + Random Forest (soft-voting ensemble)
# Classes: 0=BPSK 1=QPSK 2=FM 3=Tone-Jam 4=Sweep-Jam 5=Pulsed-Jam
# =========================================

import os
os.chdir(r'E:\D\projectsml')   # uncomment and set your path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  Multi-Class RF Signal Classifier — Confusion Matrix Only")
print("=" * 60)

# =========================================
# STEP 1: LOAD DATASET
# =========================================
df = pd.read_csv('rf_final_hybrid_dataset.csv')

FEATURE_COLS = ['PAPR_dB', 'Variance', 'Spectral_Flatness', 'Kurtosis',
                'Spectral_Entropy', 'Skewness', 'ZCR',
                'Spectral_Peak_Count', 'Envelope_Duty_Cycle',
                'Echo_Autocorr_Strength', 'Order2_PAR', 'Order4_PAR', 'Phase_Variance']

LABEL_NAMES = {0: 'BPSK', 1: 'QPSK', 2: 'FM',
               3: 'Tone-Jam', 4: 'Sweep-Jam', 5: 'Pulsed-Jam',
               6: 'Spoofed-Replay', 7: 'Gaussian-Jam'}
CLASS_ORDER = [0, 1, 2, 3, 4, 5, 6, 7]
CLASS_LABELS = [LABEL_NAMES[c] for c in CLASS_ORDER]

X = df[FEATURE_COLS].values
y = df['Label'].values

print(f"\n  [LOAD] Dataset loaded — {len(df)} samples, {X.shape[1]} features")

empty_classes = [LABEL_NAMES[c] for c in CLASS_ORDER if sum(y == c) == 0]
if empty_classes:
    raise SystemExit(
        f"\n  [ERROR] These classes have 0 samples in rf_final_hybrid_dataset.csv: "
        f"{empty_classes}\n"
        f"  Run, in order:\n"
        f"    python load_real_jamming_dataset.py\n"
        f"    python merge_final_hybrid_dataset.py\n"
        f"  then re-run this script."
    )

# =========================================
# STEP 2: TRAIN / TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================
# STEP 3: SCALING
# =========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================
# STEP 4: TRAIN SVM
# =========================================
svm_model = SVC(kernel='rbf', C=10, gamma='scale',
                 probability=True, random_state=42, class_weight='balanced')
svm_model.fit(X_train_scaled, y_train)
svm_proba = svm_model.predict_proba(X_test_scaled)

# =========================================
# STEP 5: TRAIN RANDOM FOREST
# =========================================
rf_model = RandomForestClassifier(n_estimators=150, max_depth=12,
                                   random_state=42, n_jobs=-1, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)
rf_proba = rf_model.predict_proba(X_test_scaled)

# =========================================
# STEP 6: ENSEMBLE (probability averaging) -> CONFUSION MATRIX
# =========================================
ensemble_proba = (svm_proba + rf_proba) / 2.0
ensemble_pred = np.argmax(ensemble_proba, axis=1)

ens_accuracy = accuracy_score(y_test, ensemble_pred)
ens_cm = confusion_matrix(y_test, ensemble_pred, labels=CLASS_ORDER)

print(f"\n  Ensemble Accuracy: {ens_accuracy*100:.2f}%")
print(f"\n  Confusion Matrix ({', '.join(CLASS_LABELS)}):")
print(ens_cm)

# ---- everything below this point (malicious detection log, ROC curves,
# feature importance, per-class recall panel, summary metrics panel,
# CSV export, model saving) has been commented out ----
#
# print(f"\n  MALICIOUS SIGNAL DETECTION LOG ...")
# y_test_bin = label_binarize(y_test, classes=CLASS_ORDER)
# fpr, tpr, roc_auc = {}, {}, {}
# for i, c in enumerate(CLASS_ORDER):
#     fpr[c], tpr[c], _ = roc_curve(y_test_bin[:, i], ensemble_proba[:, i])
#     roc_auc[c] = auc(fpr[c], tpr[c])
# importances = rf_model.feature_importances_
# ... (see original script for the full versions of these blocks)

# =========================================
# STEP 7: VISUALIZATION — CONFUSION MATRIX ONLY
# =========================================
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(ens_cm, interpolation='nearest', cmap='Reds')
ax.set_title(f'Confusion Matrix\nEnsemble Accuracy: {ens_accuracy*100:.1f}%', fontsize=13)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_xticks(range(len(CLASS_ORDER))); ax.set_yticks(range(len(CLASS_ORDER)))
ax.set_xticklabels(CLASS_LABELS, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(CLASS_LABELS, fontsize=9)
for i in range(len(CLASS_ORDER)):
    for j in range(len(CLASS_ORDER)):
        ax.text(j, i, str(ens_cm[i, j]), ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='white' if ens_cm[i, j] > ens_cm.max()/2 else 'black')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('confusion_matrix_only.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  [PLOT] Saved: confusion_matrix_only.png")

# =========================================
# STEP 8/9: EXPORT + MODEL SAVING — commented out
# =========================================
# pd.DataFrame(results).to_csv('multiclass_results.csv', index=False)
# joblib.dump(svm_model, 'svm_model.pkl')
# joblib.dump(rf_model, 'rf_model.pkl')
# joblib.dump(scaler, 'feature_scaler.pkl')

print("\n" + "=" * 60)
print("  Done — confusion matrix only")
print("=" * 60)