# =========================================
# Multi-Class RF Signal Classifier
# Project: RF Signal Classification and Malicious Signal Detection
# Input  : rf_multiclass_dataset.csv
# Output : multiclass_results.csv, multiclass_output.png
# Models : SVM + Random Forest (soft-voting ensemble)
# Classes: 0=BPSK 1=QPSK 2=FM 3=Tone-Jam 4=Sweep-Jam 5=Pulsed-Jam
# =========================================

import os
os.chdir(r'D:\projectsml')   # uncomment and set your path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  Multi-Class RF Signal Classifier")
print("  Signal Type + Jamming Type Detection")
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
MALICIOUS_LABELS = {3, 4, 5, 6, 7}  # jammers + spoofing = malicious
CLASS_LABELS = [LABEL_NAMES[c] for c in CLASS_ORDER]

X = df[FEATURE_COLS].values
y = df['Label'].values

print(f"\n  [LOAD] Dataset loaded")
print(f"         Total samples : {len(df)}")
for c in CLASS_ORDER:
    print(f"         {LABEL_NAMES[c]:<12}: {sum(y == c)}")
print(f"         Features      : {X.shape[1]}")

empty_classes = [LABEL_NAMES[c] for c in CLASS_ORDER if sum(y == c) == 0]
if empty_classes:
    raise SystemExit(
        f"\n  [ERROR] These classes have 0 samples in rf_final_hybrid_dataset.csv: "
        f"{empty_classes}\n"
        f"  This usually means the merge pipeline wasn't re-run after adding a new "
        f"data source. Run, in order:\n"
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
print(f"\n  [SPLIT] Train: {len(X_train)}  Test: {len(X_test)}")

# =========================================
# STEP 3: SCALING
# =========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"\n  [SCALE] StandardScaler applied")

# =========================================
# STEP 4: TRAIN SVM (multi-class via one-vs-rest, built into SVC)
# =========================================
print(f"\n  [SVM]  Training multi-class SVM...")
svm_model = SVC(kernel='rbf', C=10, gamma='scale',
                 probability=True, random_state=42, class_weight='balanced')
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_proba = svm_model.predict_proba(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_cv = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)
print(f"  [SVM]  Test Accuracy: {svm_accuracy*100:.2f}%  "
      f"CV: {svm_cv.mean()*100:.2f}% (±{svm_cv.std()*100:.2f}%)")

# =========================================
# STEP 5: TRAIN RANDOM FOREST
# =========================================
print(f"\n  [RF]   Training multi-class Random Forest...")
rf_model = RandomForestClassifier(n_estimators=150, max_depth=12,
                                   random_state=42, n_jobs=-1, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_cv = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
print(f"  [RF]   Test Accuracy: {rf_accuracy*100:.2f}%  "
      f"CV: {rf_cv.mean()*100:.2f}% (±{rf_cv.std()*100:.2f}%)")

# =========================================
# STEP 6: ENSEMBLE (probability averaging, multi-class)
# =========================================
print(f"\n  [ENS]  Probability Averaging Ensemble...")
ensemble_proba = (svm_proba + rf_proba) / 2.0
ensemble_pred = np.argmax(ensemble_proba, axis=1)

ens_accuracy = accuracy_score(y_test, ensemble_pred)
ens_precision = precision_score(y_test, ensemble_pred, average='macro')
ens_recall = recall_score(y_test, ensemble_pred, average='macro')
ens_f1 = f1_score(y_test, ensemble_pred, average='macro')
ens_cm = confusion_matrix(y_test, ensemble_pred, labels=CLASS_ORDER)

print(f"\n  {'='*50}")
print(f"  FINAL ENSEMBLE RESULTS (macro-averaged)")
print(f"  {'='*50}")
print(f"  Accuracy   : {ens_accuracy*100:.2f}%")
print(f"  Precision  : {ens_precision*100:.2f}%")
print(f"  Recall     : {ens_recall*100:.2f}%")
print(f"  F1 Score   : {ens_f1*100:.2f}%")
print(f"\n  Classification Report:")
print(classification_report(y_test, ensemble_pred,
      target_names=CLASS_LABELS, labels=CLASS_ORDER))

# =========================================
# STEP 6b: MALICIOUS DETECTION ALERTS + SIGNAL FOOTPRINT
# For every test sample the ensemble flags as malicious (jammer or
# spoofed), print an alert with the signal's feature "footprint" -
# the specific values that led to that classification.
# =========================================
print(f"\n  {'='*50}")
print(f"  MALICIOUS SIGNAL DETECTION LOG")
print(f"  {'='*50}")

X_test_raw = X_test  # unscaled features, for human-readable footprint values
malicious_count = 0

for idx in range(len(y_test)):
    pred_label = ensemble_pred[idx]
    if pred_label in MALICIOUS_LABELS:
        malicious_count += 1
        confidence = ensemble_proba[idx, pred_label] * 100
        footprint = X_test_raw[idx]

        print(f"\n  [ALERT #{malicious_count}] MALICIOUS SIGNAL PRESENT")
        print(f"  {'-'*46}")
        print(f"  Type            : {LABEL_NAMES[pred_label]}")
        print(f"  Confidence      : {confidence:.2f}%")
        print(f"  Signal Footprint:")
        for fname, fval in zip(FEATURE_COLS, footprint):
            print(f"      {fname:<22}: {fval:.4f}")

        # Type-specific interpretation of the footprint
        if pred_label == 3:
            print(f"  Interpretation  : Constant-power tone jammer - "
                  f"low Spectral_Peak_Count (~1), high Envelope_Duty_Cycle (~1.0)")
        elif pred_label == 4:
            print(f"  Interpretation  : Sweep/chirp jammer - "
                  f"elevated Spectral_Peak_Count, continuous duty cycle")
        elif pred_label == 5:
            print(f"  Interpretation  : Pulsed jammer - "
                  f"low Envelope_Duty_Cycle (bursty on/off pattern)")
        elif pred_label == 6:
            print(f"  Interpretation  : Spoofed/replayed signal - "
                  f"anomalous PAPR/Variance vs legitimate baseline, "
                  f"echo signature present")
        elif pred_label == 7:
            print(f"  Interpretation  : Gaussian-noise (barrage) jammer - "
                  f"wideband noise raises Variance/Envelope_Duty_Cycle "
                  f"while flattening Spectral_Flatness across the whole band")

        if malicious_count >= 10:
            remaining = sum(1 for p in ensemble_pred[idx+1:] if p in MALICIOUS_LABELS)
            if remaining > 0:
                print(f"\n  ... ({remaining} more malicious detections in this test set, "
                      f"log truncated for readability)")
            break

print(f"\n  {'='*50}")
print(f"  Total malicious signals detected in test set: "
      f"{sum(1 for p in ensemble_pred if p in MALICIOUS_LABELS)} / {len(y_test)}")
print(f"  {'='*50}")

# Multi-class ROC: one-vs-rest, macro-average AUC
y_test_bin = label_binarize(y_test, classes=CLASS_ORDER)
fpr, tpr, roc_auc = {}, {}, {}
for i, c in enumerate(CLASS_ORDER):
    fpr[c], tpr[c], _ = roc_curve(y_test_bin[:, i], ensemble_proba[:, i])
    roc_auc[c] = auc(fpr[c], tpr[c])
macro_auc = np.mean(list(roc_auc.values()))
print(f"  Macro-average ROC-AUC: {macro_auc:.4f}")

# Feature importance from RF
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
print(f"\n  Feature Importance (Random Forest):")
for i in sorted_idx:
    print(f"  {FEATURE_COLS[i]:<22}: {importances[i]*100:.2f}%")

# =========================================
# STEP 7: VISUALIZATION
# =========================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    'Multi-Class RF Signal Classifier — Signal Type + Jamming Type Detection\n'
    'SVM + Random Forest Ensemble',
    fontsize=13, fontweight='bold'
)

# --- Panel 1: Confusion Matrix (6x6) ---
ax = axes[0][0]
im = ax.imshow(ens_cm, interpolation='nearest', cmap='Reds')
ax.set_title(f'Confusion Matrix\nEnsemble Accuracy: {ens_accuracy*100:.1f}%', fontsize=11)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_xticks(range(len(CLASS_ORDER))); ax.set_yticks(range(len(CLASS_ORDER)))
ax.set_xticklabels(CLASS_LABELS, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(CLASS_LABELS, fontsize=8)
for i in range(len(CLASS_ORDER)):
    for j in range(len(CLASS_ORDER)):
        ax.text(j, i, str(ens_cm[i, j]), ha='center', va='center',
                fontsize=9, fontweight='bold',
                color='white' if ens_cm[i, j] > ens_cm.max()/2 else 'black')
plt.colorbar(im, ax=ax)

# --- Panel 2: Multi-class ROC (one-vs-rest) ---
ax = axes[0][1]
colors = plt.cm.tab10(np.linspace(0, 1, len(CLASS_ORDER)))
for c, color in zip(CLASS_ORDER, colors):
    ax.plot(fpr[c], tpr[c], color=color, linewidth=1.5,
            label=f'{LABEL_NAMES[c]} (AUC={roc_auc[c]:.3f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
ax.set_title(f'ROC Curves (One-vs-Rest)\nMacro AUC: {macro_auc:.4f}', fontsize=11)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right', fontsize=7)
ax.grid(True, alpha=0.3)

# --- Panel 3: Model Accuracy Comparison ---
ax = axes[0][2]
models = ['SVM', 'Random\nForest', 'Ensemble\n(Avg)']
acc = [svm_accuracy*100, rf_accuracy*100, ens_accuracy*100]
bars = ax.bar(models, acc, color=['steelblue', 'forestgreen', 'crimson'],
              edgecolor='black', width=0.5)
ax.set_title('Model Accuracy Comparison', fontsize=11)
ax.set_ylabel('Accuracy (%)')
ax.set_ylim([0, 105])
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, acc):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

# --- Panel 4: Feature Importance ---
ax = axes[1][0]
names_sorted = [FEATURE_COLS[i] for i in sorted_idx]
vals_sorted = [importances[i]*100 for i in sorted_idx]
ax.barh(names_sorted, vals_sorted, color='forestgreen', edgecolor='black')
ax.set_title('Feature Importance (Random Forest)', fontsize=11)
ax.set_xlabel('Importance (%)')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(vals_sorted):
    ax.text(v+0.3, i, f'{v:.1f}%', va='center', fontsize=8)

# --- Panel 5: Per-class accuracy (recall) ---
ax = axes[1][1]
per_class_recall = ens_cm.diagonal() / ens_cm.sum(axis=1)
bar_colors = ['steelblue' if c not in MALICIOUS_LABELS else 'crimson' for c in CLASS_ORDER]
ax.bar(CLASS_LABELS, per_class_recall*100,
       color=bar_colors, edgecolor='black')
ax.set_title('Per-Class Recall\n(Blue=Normal types, Red=Jammer types)', fontsize=11)
ax.set_ylabel('Recall (%)')
ax.set_ylim([0, 105])
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# --- Panel 6: Summary metrics ---
ax = axes[1][2]
metrics = ['Accuracy', 'Precision\n(macro)', 'Recall\n(macro)', 'F1\n(macro)']
metric_vals = [ens_accuracy, ens_precision, ens_recall, ens_f1]
bars2 = ax.bar(metrics, [v*100 for v in metric_vals],
               color=['crimson', 'darkorange', 'steelblue', 'forestgreen'],
               edgecolor='black', width=0.5)
ax.set_title('Ensemble Performance Metrics (Macro-Avg)', fontsize=11)
ax.set_ylabel('Score (%)')
ax.set_ylim([0, 105])
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars2, metric_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('multiclass_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  [PLOT] Saved: multiclass_output.png")

# =========================================
# STEP 8: EXPORT RESULTS
# =========================================
results = {
    'Model': ['SVM', 'Random Forest', 'Ensemble'],
    'Accuracy': [round(svm_accuracy, 4), round(rf_accuracy, 4), round(ens_accuracy, 4)],
    'Precision_macro': [
        round(precision_score(y_test, svm_pred, average='macro'), 4),
        round(precision_score(y_test, rf_pred, average='macro'), 4),
        round(ens_precision, 4)],
    'Recall_macro': [
        round(recall_score(y_test, svm_pred, average='macro'), 4),
        round(recall_score(y_test, rf_pred, average='macro'), 4),
        round(ens_recall, 4)],
    'F1_macro': [
        round(f1_score(y_test, svm_pred, average='macro'), 4),
        round(f1_score(y_test, rf_pred, average='macro'), 4),
        round(ens_f1, 4)],
}
pd.DataFrame(results).to_csv('multiclass_results.csv', index=False)
print("  [CSV]  Saved: multiclass_results.csv")

print("\n" + "=" * 60)
print("  Multi-Class Classification Complete")
print("=" * 60)

# =========================================
# STEP 9: SAVE TRAINED MODELS FOR REUSE
# So other scripts (like the final dashboard) can classify a NEW
# captured signal instead of just reporting these aggregate test stats.
# =========================================
import joblib
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
print("  [SAVE]  svm_model.pkl, rf_model.pkl, feature_scaler.pkl saved")
print("          (used by result.py to classify new captured signals)")