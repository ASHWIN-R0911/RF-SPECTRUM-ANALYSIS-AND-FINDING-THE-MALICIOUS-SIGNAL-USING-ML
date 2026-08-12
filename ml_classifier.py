# =========================================
# Dual-ML Classifier
# Project: Malicious RF Signal Detection
# Input  : features_dataset.csv
# Output : classification_results.csv
# Models : SVM + Random Forest (ensemble)
# Decision: Probability Averaging
# =========================================

import os
os.chdir(r'D:\projectsml')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report
)
import csv
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  Dual-ML Classifier — SVM + Random Forest Ensemble")
print("=" * 60)

# =========================================
# STEP 1: LOAD FEATURE DATASET
# =========================================
df = pd.read_csv('rf_jamming_dataset.csv')

# Features and labels
X = df[['PAPR_dB', 'Variance', 'Spectral_Flatness', 'Kurtosis',
              'Spectral_Entropy', 'Skewness', 'ZCR']].values
y = df['Label'].values

print(f"\n  [LOAD] Dataset loaded")
print(f"         Total samples   : {len(df)}")
print(f"         Malicious (1)   : {sum(y==1)}")
print(f"         Normal    (0)   : {sum(y==0)}")
print(f"         Features        : {X.shape[1]}")
print(f"         Feature names   : PAPR | Variance |  Spectral Flatness | Kurtosis | spectral Entropy | Skewness | ZCR")
print(f"                           ")

# =========================================
# STEP 2: TRAIN / TEST SPLIT
# 80% training, 20% testing
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  [SPLIT] Train/Test Split (80/20)")
print(f"          Training samples : {len(X_train)}")
print(f"          Testing samples  : {len(X_test)}")

# =========================================
# STEP 3: FEATURE SCALING
# SVM requires scaled features
# RF doesn't require it but benefits from it
# =========================================
scaler   = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n  [SCALE] StandardScaler applied")

# =========================================
# STEP 4: TRAIN SVM
# =========================================
print(f"\n  [SVM]  Training SVM classifier...")
svm_model = SVC(
    kernel='rbf',         # RBF kernel for non-linear separation
    C=10,                 # Regularization — higher = tighter fit
    gamma='scale',        # Auto-scale gamma to features
    probability=True,     # Enable probability output for averaging
    random_state=42
)
svm_model.fit(X_train_scaled, y_train)

# SVM predictions and probabilities
svm_pred       = svm_model.predict(X_test_scaled)
svm_proba      = svm_model.predict_proba(X_test_scaled)[:, 1]
svm_accuracy   = accuracy_score(y_test, svm_pred)

# Cross-validation score
svm_cv = cross_val_score(svm_model, X_train_scaled,
                          y_train, cv=5, scoring='accuracy')

print(f"  [SVM]  Training complete")
print(f"         Test Accuracy    : {svm_accuracy*100:.2f}%")
print(f"         CV Accuracy      : {svm_cv.mean()*100:.2f}% "
      f"(±{svm_cv.std()*100:.2f}%)")

# =========================================
# STEP 5: TRAIN RANDOM FOREST
# =========================================
print(f"\n  [RF]   Training Random Forest classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,     # 100 decision trees
    max_depth=10,         # Max tree depth
    min_samples_split=2,
    random_state=42,
    n_jobs=-1             # Use all CPU cores
)
rf_model.fit(X_train_scaled, y_train)

# RF predictions and probabilities
rf_pred      = rf_model.predict(X_test_scaled)
rf_proba     = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_accuracy  = accuracy_score(y_test, rf_pred)

# Cross-validation score
rf_cv = cross_val_score(rf_model, X_train_scaled,
                         y_train, cv=5, scoring='accuracy')

print(f"  [RF]   Training complete")
print(f"         Test Accuracy    : {rf_accuracy*100:.2f}%")
print(f"         CV Accuracy      : {rf_cv.mean()*100:.2f}% "
      f"(±{rf_cv.std()*100:.2f}%)")

# =========================================
# STEP 6: PROBABILITY AVERAGING ENSEMBLE
# Final score = average of SVM + RF proba
# Threshold = 0.5
# =========================================
print(f"\n  [ENS]  Probability Averaging Ensemble...")
ensemble_proba = (svm_proba + rf_proba) / 2.0
threshold      = 0.5
ensemble_pred  = (ensemble_proba >= threshold).astype(int)
ens_accuracy   = accuracy_score(y_test, ensemble_pred)
ens_precision  = precision_score(y_test, ensemble_pred)
ens_recall     = recall_score(y_test, ensemble_pred)
ens_f1         = f1_score(y_test, ensemble_pred)
ens_cm         = confusion_matrix(y_test, ensemble_pred)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, ensemble_proba)
roc_auc     = auc(fpr, tpr)

print(f"\n  {'='*50}")
print(f"  FINAL ENSEMBLE RESULTS")
print(f"  {'='*50}")
print(f"  Accuracy        : {ens_accuracy*100:.2f}%")
print(f"  Precision       : {ens_precision*100:.2f}%")
print(f"  Recall          : {ens_recall*100:.2f}%")
print(f"  F1 Score        : {ens_f1*100:.2f}%")
print(f"  ROC-AUC         : {roc_auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"  TN={ens_cm[0,0]}  FP={ens_cm[0,1]}")
print(f"  FN={ens_cm[1,0]}  TP={ens_cm[1,1]}")
print(f"\n  Classification Report:")
print(classification_report(y_test, ensemble_pred,
      target_names=['Normal (0)', 'Malicious (1)']))

# Feature importance from RF
feat_names = ['PAPR_dB', 'Variance', 'Spectral_Flatness', 'Kurtosis',
              'Spectral_Entropy', 'Skewness', 'ZCR']
importances = rf_model.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]

print(f"  Feature Importance (Random Forest):")
for i in sorted_idx:
    print(f"  {feat_names[i]:<20}: {importances[i]*100:.2f}%")

# =========================================
# STEP 7: VISUALIZATION — 6 PANELS
# =========================================
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
fig.suptitle(
    'Dual-ML Classifier Results — SVM + Random Forest Ensemble\n'
    'Malicious RF Signal Detection | 433.92 MHz ISM Band',
    fontsize=13, fontweight='bold'
)

# --- Panel 1: Confusion Matrix ---
ax = axes[0][0]
cm_display = ens_cm
im = ax.imshow(cm_display, interpolation='nearest', cmap='Reds')
ax.set_title(f'Confusion Matrix\nEnsemble Accuracy: '
             f'{ens_accuracy*100:.1f}%', fontsize=11)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Normal (0)', 'Malicious (1)'])
ax.set_yticklabels(['Normal (0)', 'Malicious (1)'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm_display[i, j]),
                ha='center', va='center',
                fontsize=16, fontweight='bold',
                color='white' if cm_display[i,j] > cm_display.max()/2
                else 'black')
plt.colorbar(im, ax=ax)

# --- Panel 2: ROC Curve ---
ax = axes[0][1]
ax.plot(fpr, tpr, color='red', linewidth=2,
        label=f'Ensemble ROC (AUC = {roc_auc:.4f})')
ax.plot([0,1], [0,1], color='gray',
        linestyle='--', linewidth=1, label='Random Baseline')
ax.fill_between(fpr, tpr, alpha=0.15, color='red')
ax.set_title('ROC Curve — Ensemble Classifier', fontsize=11)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate (Recall)')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

# --- Panel 3: Model Accuracy Comparison ---
ax = axes[0][2]
models  = ['SVM', 'Random\nForest', 'Ensemble\n(Avg)']
acc     = [svm_accuracy*100, rf_accuracy*100, ens_accuracy*100]
colors  = ['steelblue', 'forestgreen', 'crimson']
bars    = ax.bar(models, acc, color=colors,
                 edgecolor='black', width=0.5)
ax.set_title('Model Accuracy Comparison', fontsize=11)
ax.set_ylabel('Accuracy (%)')
ax.set_ylim([80, 105])
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, acc):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center',
            va='bottom', fontweight='bold', fontsize=11)

# --- Panel 4: Feature Importance (RF) ---
ax = axes[1][0]
feat_imp_sorted = [(feat_names[i], importances[i])
                   for i in sorted_idx]
names_sorted = [x[0] for x in feat_imp_sorted]
vals_sorted  = [x[1]*100 for x in feat_imp_sorted]
ax.barh(names_sorted, vals_sorted,
        color='forestgreen', edgecolor='black')
ax.set_title('Feature Importance\n(Random Forest)', fontsize=11)
ax.set_xlabel('Importance (%)')
ax.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(vals_sorted):
    ax.text(v + 0.3, i, f'{v:.1f}%',
            va='center', fontsize=9)

# --- Panel 5: Probability Distribution ---
ax = axes[1][1]
ax.scatter(ensemble_proba[y_test==0],
           np.zeros(sum(y_test==0)),
           color='blue', s=200, label='Normal (0)',
           zorder=5, marker='o')
ax.scatter(ensemble_proba[y_test==1],
           np.ones(sum(y_test==1)),
           color='red', s=200, label='Malicious (1)',
           zorder=5, marker='o')
ax.axvline(x=threshold, color='black',
           linewidth=2, linestyle='--',
           label=f'Decision Threshold: {threshold}')
ax.set_title('Ensemble Probability Distribution\n'
             'Malicious vs Normal', fontsize=11)
ax.set_xlabel('P(Malicious)')
ax.set_ylabel('Class')
ax.set_yticks([0, 1])
ax.set_yticklabels(['Normal (0)', 'Malicious (1)'])
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.5, 1.5])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Panel 6: Metrics Summary Bar ---
ax = axes[1][2]
metrics       = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metric_vals   = [ens_accuracy, ens_precision,
                 ens_recall, ens_f1]
metric_colors = ['crimson', 'darkorange',
                 'steelblue', 'forestgreen']
bars2 = ax.bar(metrics, [v*100 for v in metric_vals],
               color=metric_colors,
               edgecolor='black', width=0.5)
ax.set_title('Ensemble Performance Metrics', fontsize=11)
ax.set_ylabel('Score (%)')
ax.set_ylim([80, 105])
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars2, metric_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{val*100:.1f}%', ha='center',
            va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('ml_classifier_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  [PLOT] Saved: ml_classifier_output.png")

# =========================================
# STEP 8: EXPORT RESULTS TO CSV
# =========================================
results = {
    'Model'        : ['SVM', 'Random Forest', 'Ensemble'],
    'Accuracy'     : [round(svm_accuracy,4),
                      round(rf_accuracy,4),
                      round(ens_accuracy,4)],
    'Precision'    : [round(precision_score(y_test,svm_pred),4),
                      round(precision_score(y_test,rf_pred),4),
                      round(ens_precision,4)],
    'Recall'       : [round(recall_score(y_test,svm_pred),4),
                      round(recall_score(y_test,rf_pred),4),
                      round(ens_recall,4)],
    'F1_Score'     : [round(f1_score(y_test,svm_pred),4),
                      round(f1_score(y_test,rf_pred),4),
                      round(ens_f1,4)],
    'ROC_AUC'      : [round(auc(*roc_curve(y_test,svm_proba)[:2]),4),
                      round(auc(*roc_curve(y_test,rf_proba)[:2]),4),
                      round(roc_auc,4)]
}
pd.DataFrame(results).to_csv(
    'classification_results.csv', index=False
)
print("  [CSV]  Saved: classification_results.csv")

print("\n" + "=" * 60)
print("  ML Classification Complete")
print("  Next Step: final_output.py")
print("=" * 60)