"""
Multi-Class RF Signal Classifier - HYBRID DATASET version
Input   : rf_final_hybrid_dataset.csv  (RadioML BPSK/QPSK/FM + Kaggle RSSI Tone/Gaussian
          + NumPy-synthetic Sweep/Pulsed/Spoofed-Replay; 13 features)
Split   : random stratified 80/20 (seed 42)  - reproduces SVM 99.4 / RF 99.8 / ensemble 99.8
Output  : multiclass_hybrid_output.png, multiclass_hybrid_results.csv,
          ensemble_bundle_hybrid.pkl, svm_model_hybrid.pkl, rf_model_hybrid.pkl,
          feature_scaler_hybrid.pkl   (copy over svm_model.pkl / rf_model.pkl /
          feature_scaler.pkl if result.py / classify_live.py should use them)

NOTE for the paper: this is a random split of a dataset whose classes come from
different data sources, so the accuracy describes this dataset, not real-world
over-the-air generalisation. Say so in the Dataset / Limitations text.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold, train_test_split,
                                     cross_val_score, cross_val_predict)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)

warnings.filterwarnings("ignore")

# =========================================
# CONFIG
# =========================================
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
DATA_DIR = os.environ.get("RF_DATA_DIR", _HERE)

INPUT_CSV = "rf_final_hybrid_dataset.csv"          # hybrid dataset (6,257 rows)
GROUP_COL = "Group_ID"        # one id per source recording; optional column
REJECT_THRESHOLD = None       # None = choose automatically from out-of-fold TRAIN
                              # predictions (test set is never used for this).
                              # Or set a number, e.g. 0.60.
TARGET_SELECTIVE_ACC = 0.99   # auto mode: lowest threshold whose accepted-sample
                              # accuracy reaches this (= highest coverage) ...
MIN_COVERAGE = 0.90           # ... while keeping at least this coverage
RANDOM_STATE = 42
SHOW_PLOT = True

FEATURE_COLS = ['PAPR_dB', 'Variance', 'Spectral_Flatness', 'Kurtosis',
                'Spectral_Entropy', 'Skewness', 'ZCR',
                'Spectral_Peak_Count', 'Envelope_Duty_Cycle',
                'Echo_Autocorr_Strength', 'Order2_PAR', 'Order4_PAR',
                'Phase_Variance']

LABEL_NAMES = {0: 'BPSK', 1: 'QPSK', 2: 'FM',
               3: 'Tone-Jam', 4: 'Sweep-Jam', 5: 'Pulsed-Jam',
               6: 'Spoofed-Replay', 7: 'Gaussian-Jam'}
CLASS_ORDER = [0, 1, 2, 3, 4, 5, 6, 7]
MALICIOUS_LABELS = {3, 4, 5, 6, 7}          # jammers + spoofing
CLASS_LABELS = [LABEL_NAMES[c] for c in CLASS_ORDER]


def path(name):
    return os.path.join(DATA_DIR, name)


def wilson_lower(k, n, z=1.96):
    """Lower end of the Wilson 95% interval for k successes out of n."""
    if n == 0:
        return float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / d


print("=" * 60)
print("  Multi-Class RF Signal Classifier (v2)")
print("  Signal Type + Jamming Type Detection (8 classes)")
print("=" * 60)

# =========================================
# STEP 1: LOAD
# =========================================
df = pd.read_csv(path(INPUT_CSV))
missing = [c for c in FEATURE_COLS + ["Label"] if c not in df.columns]
if missing:
    raise SystemExit(f"  [ERROR] Missing columns in {INPUT_CSV}: {missing}")

X = df[FEATURE_COLS].values
y = df["Label"].values.astype(int)
has_groups = False    # hybrid CSV has no Group_ID -> random stratified 80/20 split
groups = df[GROUP_COL].values if has_groups else None

print(f"\n  [LOAD] {len(df)} samples, {X.shape[1]} features")
for c in CLASS_ORDER:
    print(f"         {LABEL_NAMES[c]:<15}: {int((y == c).sum())}")

unknown = sorted(set(y) - set(CLASS_ORDER))
if unknown:
    raise SystemExit(f"  [ERROR] Labels outside 0-7 found: {unknown}")

if has_groups:
    if df[GROUP_COL].isna().any() or (df[GROUP_COL].astype(str).str.strip() == "").any():
        raise SystemExit(f"  [ERROR] {GROUP_COL} has missing/blank values. "
                         "Fix the dataset instead of guessing groups.")
    gdf = pd.DataFrame({"g": df[GROUP_COL].astype(str), "y": y})
    n_groups = gdf["g"].nunique()
    mixed = gdf.groupby("g")["y"].nunique()
    mixed = mixed[mixed > 1]
    print(f"\n  [GROUPS] {n_groups} groups for {len(df)} rows "
          f"(median {int(gdf.groupby('g').size().median())} rows/group)")
    if len(mixed):
        raise SystemExit(f"  [ERROR] {len(mixed)} group(s) contain more than one class "
                         f"(e.g. {list(mixed.index[:3])}). A group must be ONE source "
                         "recording, so Group_ID is wrongly generated.")
    if n_groups > 0.9 * len(df):
        print("  [NOTE] Almost every row has its own Group_ID. That is correct ONLY if "
              "each row is an independent draw (e.g. its own random seed / its own "
              "capture file). If rows are windows cut from shared recordings, "
              "group splitting does NOT stop leakage - fix Group_ID.")
    per_class_groups = gdf.groupby("y")["g"].nunique()
    print("         groups per class: " + ", ".join(
        f"{LABEL_NAMES[c]}={int(per_class_groups.get(c, 0))}" for c in CLASS_ORDER))
    low = [LABEL_NAMES[c] for c in CLASS_ORDER if per_class_groups.get(c, 0) < 3]
    if low:
        raise SystemExit(f"  [ERROR] Fewer than 3 groups for: {low}. Cannot split or "
                         "cross-validate by group. Generate more independent recordings.")

empty = [LABEL_NAMES[c] for c in CLASS_ORDER if (y == c).sum() == 0]
if empty:
    raise SystemExit(
        f"\n  [ERROR] Classes with 0 samples: {empty}\n"
        "  Re-run load_real_jamming_dataset.py and merge_final_hybrid_dataset.py, "
        "then this script.")

# =========================================
# STEP 2: SPLIT (group-aware when possible)
# =========================================
if has_groups:
    split_mode = "GROUP"
    for seed in range(RANDOM_STATE, RANDOM_STATE + 200):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, te = next(gss.split(X, y, groups))
        if set(y[tr]) == set(CLASS_ORDER) and set(y[te]) == set(CLASS_ORDER):
            break
    else:
        raise SystemExit("  [ERROR] Could not find a group split with every class "
                         "in both train and test. Add more groups per class.")
    assert not (set(groups[tr]) & set(groups[te]))
    print(f"\n  [SPLIT] GROUP split on '{GROUP_COL}' (seed {seed}): "
          f"train {len(tr)} / test {len(te)}; "
          f"{len(set(groups[tr]))} / {len(set(groups[te]))} groups")
else:
    split_mode = "random stratified 80/20"
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                              random_state=RANDOM_STATE, stratify=y)
    print(f"\n  [WARN] No '{GROUP_COL}' column in {INPUT_CSV}.")
    print("         Using a RANDOM split: windows cut from the same recording can")
    print("         land in both train and test, so accuracy can be too high.")
    print("         Add Group_ID in merge_final_hybrid_dataset.py to fix this.")
    print(f"  [SPLIT] train {len(tr)} / test {len(te)}")

X_train, X_test, y_train, y_test = X[tr], X[te], y[tr], y[te]
g_train = groups[tr] if has_groups else None

if has_groups:
    n_splits = min(5, int(min(len(set(g_train[y_train == c])) for c in CLASS_ORDER)))
    if n_splits < 2:
        raise SystemExit("  [ERROR] Fewer than 2 training groups for some class; "
                         "cannot run group cross-validation.")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_kwargs = {"groups": g_train}
else:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_kwargs = {}

# every CV fold must contain all 8 classes in BOTH train and validation part
cv_problems = []
for k, (a, b) in enumerate(cv.split(X_train, y_train, g_train)):
    miss_a = sorted(set(CLASS_ORDER) - set(y_train[a]))
    miss_b = sorted(set(CLASS_ORDER) - set(y_train[b]))
    if miss_a or miss_b:
        cv_problems.append((k, miss_a, miss_b))
cv_all_classes = not cv_problems
if cv_all_classes:
    print(f"  [CV]   {type(cv).__name__}: all 8 classes present in every fold "
          f"(train and validation)")
else:
    print("  [WARN] Some CV folds miss classes (fold, missing-in-train, missing-in-val):")
    for p in cv_problems:
        print(f"         {p}")
    print("         CV numbers are less reliable. Report this limitation, or add "
          "more groups for those classes.")

# =========================================
# STEP 3-5: MODELS (scaler inside the pipeline)
# =========================================
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                random_state=RANDOM_STATE, class_weight="balanced")),
])
# Random Forest does not need scaling; it is kept on scaled inputs only so the
# legacy result.py / classify_live.py (which scale first) stay compatible.
rf_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=150, max_depth=12,
                                  random_state=RANDOM_STATE, n_jobs=-1,
                                  class_weight="balanced")),
])

results = {}
for name, pipe in [("SVM", svm_pipe), ("Random Forest", rf_pipe)]:
    print(f"\n  [{name}] cross-validating ({type(cv).__name__}) ...")
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, **cv_kwargs)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    results[name] = {"pred": pred, "proba": proba,
                     "acc": accuracy_score(y_test, pred),
                     "cv_mean": scores.mean(), "cv_std": scores.std()}
    print(f"  [{name}] Test accuracy {results[name]['acc']*100:.2f}%  |  "
          f"CV {scores.mean()*100:.2f}% (+/-{scores.std()*100:.2f}%)")

svm_pred, svm_proba = results["SVM"]["pred"], results["SVM"]["proba"]
rf_pred, rf_proba = results["Random Forest"]["pred"], results["Random Forest"]["proba"]
svm_accuracy, rf_accuracy = results["SVM"]["acc"], results["Random Forest"]["acc"]

# =========================================
# STEP 6: ENSEMBLE  (probability averaging) + reject option
# =========================================
assert list(svm_pipe.classes_) == CLASS_ORDER and list(rf_pipe.classes_) == CLASS_ORDER, \
    f"probability columns are not in class order 0-7: {svm_pipe.classes_}, {rf_pipe.classes_}"

# ---- choose the reject threshold on out-of-fold TRAIN predictions
oof = (cross_val_predict(svm_pipe, X_train, y_train, cv=cv, method="predict_proba", **cv_kwargs)
       + cross_val_predict(rf_pipe, X_train, y_train, cv=cv, method="predict_proba", **cv_kwargs)) / 2.0
oof_pred, oof_best = oof.argmax(axis=1), oof.max(axis=1)
print("\n  [REJECT] validation (out-of-fold, train only):")
print("     threshold  coverage  selective-acc")
cands = []
for th in np.arange(0.30, 0.96, 0.05):
    m = oof_best >= th
    if m.sum() == 0:
        continue
    cov, sel = m.mean(), (oof_pred[m] == y_train[m]).mean()
    cands.append((round(float(th), 2), cov, sel))
    print(f"       {th:5.2f}     {cov*100:6.2f}%    {sel*100:6.2f}%")
if REJECT_THRESHOLD is None:
    ok = [c for c in cands if c[1] >= MIN_COVERAGE and c[2] >= TARGET_SELECTIVE_ACC]
    if ok:
        REJECT_THRESHOLD = ok[0][0]
    else:
        pool = [c for c in cands if c[1] >= MIN_COVERAGE] or cands
        REJECT_THRESHOLD = max(pool, key=lambda c: (c[2], c[1]))[0]
        print(f"  [WARN] No threshold reaches {TARGET_SELECTIVE_ACC*100:.0f}% selective "
              f"accuracy at >= {MIN_COVERAGE*100:.0f}% coverage; using best available.")
    print(f"  [REJECT] Chosen threshold = {REJECT_THRESHOLD} (test set not used)")
else:
    print(f"  [REJECT] Using fixed threshold = {REJECT_THRESHOLD}")

ensemble_proba = (svm_proba + rf_proba) / 2.0
ensemble_pred = np.argmax(ensemble_proba, axis=1)      # classes are 0..7 in order
best_p = ensemble_proba.max(axis=1)
accepted = best_p >= REJECT_THRESHOLD

ens_accuracy = accuracy_score(y_test, ensemble_pred)
ens_precision = precision_score(y_test, ensemble_pred, average="macro")
ens_recall = recall_score(y_test, ensemble_pred, average="macro")
ens_f1 = f1_score(y_test, ensemble_pred, average="macro")
ens_cm = confusion_matrix(y_test, ensemble_pred, labels=CLASS_ORDER)   # 8 x 8

print(f"\n  {'='*50}")
print(f"  (1) EIGHT-CLASS PERFORMANCE BEFORE REJECTION   split = {split_mode}")
print(f"      all {len(y_test)} test samples, macro-averaged")
print(f"  {'='*50}")
print(f"  Accuracy   : {ens_accuracy*100:.2f}%")
print(f"  Precision  : {ens_precision*100:.2f}%")
print(f"  Recall     : {ens_recall*100:.2f}%")
print(f"  F1 Score   : {ens_f1*100:.2f}%")
print(f"  Ensemble vs best single model: "
      f"{(ens_accuracy - max(svm_accuracy, rf_accuracy))*100:+.2f} points")

coverage = float(accepted.mean())
if accepted.any():
    sel_acc = accuracy_score(y_test[accepted], ensemble_pred[accepted])
    sel_prec = precision_score(y_test[accepted], ensemble_pred[accepted],
                               average="macro", labels=CLASS_ORDER, zero_division=0)
    sel_rec = recall_score(y_test[accepted], ensemble_pred[accepted],
                           average="macro", labels=CLASS_ORDER, zero_division=0)
    sel_f1 = f1_score(y_test[accepted], ensemble_pred[accepted],
                      average="macro", labels=CLASS_ORDER, zero_division=0)
else:
    sel_acc = sel_prec = sel_rec = sel_f1 = float("nan")
strict_acc = float(((ensemble_pred == y_test) & accepted).mean())   # rejects count as errors
print(f"\n  {'='*50}")
print(f"  (2) SELECTIVE PERFORMANCE AFTER REJECTION (threshold {REJECT_THRESHOLD})")
print(f"  {'='*50}")
print(f"  Coverage (accepted)      : {coverage*100:.2f}%")
print(f"  Rejection rate           : {(1-coverage)*100:.2f}%  "
      f"({int((~accepted).sum())} of {len(y_test)} -> 'Uncertain')")
print(f"  Selective accuracy       : {sel_acc*100:.2f}%  (accepted samples only)")
print(f"  Selective macro P/R/F1   : {sel_prec*100:.2f}% / {sel_rec*100:.2f}% / {sel_f1*100:.2f}%")
print(f"  Accuracy, rejects = wrong: {strict_acc*100:.2f}%")

print("\n  Classification Report:")
print(classification_report(y_test, ensemble_pred, target_names=CLASS_LABELS,
                            labels=CLASS_ORDER))

print("  Per-class recall with Wilson 95% lower bound (small classes!):")
for i, c in enumerate(CLASS_ORDER):
    n_c = int(ens_cm[i].sum())
    k_c = int(ens_cm[i, i])
    print(f"     {LABEL_NAMES[c]:<15} {k_c:>4}/{n_c:<4} = {k_c/max(n_c,1)*100:6.2f}%   "
          f"lower bound {wilson_lower(k_c, n_c)*100:6.2f}%")

# =========================================
# STEP 6b: MALICIOUS ALERT LOG (evidence = unusual features)
# =========================================
scaler_fitted = svm_pipe.named_steps["scaler"]
Z_test = (X_test - scaler_fitted.mean_) / scaler_fitted.scale_

print(f"\n  {'='*50}\n  MALICIOUS SIGNAL DETECTION LOG\n  {'='*50}")
mal_idx = [i for i in range(len(y_test))
           if ensemble_pred[i] in MALICIOUS_LABELS and accepted[i]]
for n_alert, i in enumerate(mal_idx[:10], 1):
    p = ensemble_pred[i]
    top = np.argsort(-np.abs(Z_test[i]))[:3]
    print(f"\n  [ALERT #{n_alert}] {LABEL_NAMES[p]}  "
          f"(model-estimated probability {ensemble_proba[i, p]*100:.1f}%)")
    print("     Most unusual features vs training data (z-score):")
    for j in top:
        print(f"       {FEATURE_COLS[j]:<24} value {X_test[i, j]:10.4f}   z = {Z_test[i, j]:+6.2f}")
if len(mal_idx) > 10:
    print(f"\n  ... {len(mal_idx) - 10} more malicious detections not shown")
print(f"\n  Malicious detections in test set: {len(mal_idx)} / {len(y_test)} "
      f"(+ {int(((ensemble_pred >= 0) & ~accepted).sum())} rejected as Uncertain)")

# =========================================
# ROC / feature importance
# =========================================
y_test_bin = label_binarize(y_test, classes=CLASS_ORDER)
fpr, tpr, roc_auc = {}, {}, {}
for i, c in enumerate(CLASS_ORDER):
    fpr[c], tpr[c], _ = roc_curve(y_test_bin[:, i], ensemble_proba[:, i])
    roc_auc[c] = auc(fpr[c], tpr[c])
macro_auc = float(np.mean(list(roc_auc.values())))
print(f"\n  Macro-average ROC-AUC: {macro_auc:.4f}")

rf_model = rf_pipe.named_steps["rf"]
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
print("\n  Feature Importance (Random Forest):")
for i in sorted_idx:
    print(f"  {FEATURE_COLS[i]:<24}: {importances[i]*100:.2f}%")

# =========================================
# STEP 7: PLOTS
# =========================================
fig, axes = plt.subplots(2, 3, figsize=(19, 13))
fig.suptitle(f"Multi-Class RF Signal Classifier - SVM + Random Forest Ensemble "
             f"(split: {split_mode})", fontsize=13, fontweight="bold", y=0.985)

ax = axes[0][0]
im = ax.imshow(ens_cm, interpolation="nearest", cmap="Reds")
ax.set_title(f"Confusion Matrix (8x8)\nEnsemble accuracy {ens_accuracy*100:.1f}%", fontsize=11)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_xticks(range(8)); ax.set_yticks(range(8))
ax.set_xticklabels(CLASS_LABELS, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(CLASS_LABELS, fontsize=8)
for i in range(8):
    for j in range(8):
        ax.text(j, i, str(ens_cm[i, j]), ha="center", va="center", fontsize=8,
                fontweight="bold",
                color="white" if ens_cm[i, j] > ens_cm.max() / 2 else "black")
plt.colorbar(im, ax=ax)

ax = axes[0][1]
for c, color in zip(CLASS_ORDER, plt.cm.tab10(np.linspace(0, 1, 8))):
    ax.plot(fpr[c], tpr[c], color=color, lw=1.5,
            label=f"{LABEL_NAMES[c]} (AUC={roc_auc[c]:.3f})")
ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
ax.set_title(f"ROC (one-vs-rest)\nMacro AUC {macro_auc:.4f}", fontsize=11)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right", fontsize=7); ax.grid(True, alpha=0.3)

ax = axes[0][2]
vals = [svm_accuracy * 100, rf_accuracy * 100, ens_accuracy * 100]
bars = ax.bar(["SVM", "Random\nForest", "Ensemble\n(avg)"], vals,
              color=["steelblue", "forestgreen", "crimson"], edgecolor="black", width=0.5)
ax.set_title("Model Accuracy Comparison", fontsize=11); ax.set_ylabel("Accuracy (%)")
ax.set_ylim([0, 105]); ax.grid(True, alpha=0.3, axis="y")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.1f}%",
            ha="center", va="bottom", fontweight="bold")

ax = axes[1][0]
ax.barh([FEATURE_COLS[i] for i in sorted_idx], [importances[i] * 100 for i in sorted_idx],
        color="forestgreen", edgecolor="black")
ax.set_title("Feature Importance (Random Forest)", fontsize=11)
ax.set_xlabel("Importance (%)"); ax.invert_yaxis(); ax.grid(True, alpha=0.3, axis="x")

ax = axes[1][1]
recall_c = ens_cm.diagonal() / ens_cm.sum(axis=1)
lower_c = [wilson_lower(int(ens_cm[i, i]), int(ens_cm[i].sum())) for i in range(8)]
ax.bar(CLASS_LABELS, recall_c * 100,
       color=["crimson" if c in MALICIOUS_LABELS else "steelblue" for c in CLASS_ORDER],
       edgecolor="black")
ax.plot(CLASS_LABELS, np.array(lower_c) * 100, "k_", ms=18, mew=2,
        label="Wilson 95% lower bound")
ax.set_title("Per-class recall (red = malicious)", fontsize=11)
ax.set_ylabel("Recall (%)"); ax.set_ylim([0, 105]); ax.legend(fontsize=8)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

ax = axes[1][2]
mvals = [ens_accuracy, ens_precision, ens_recall, ens_f1]
bars2 = ax.bar(["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)"],
               [v * 100 for v in mvals],
               color=["crimson", "darkorange", "steelblue", "forestgreen"],
               edgecolor="black", width=0.5)
ax.set_title("Ensemble Metrics (macro-avg)", fontsize=11); ax.set_ylabel("Score (%)")
ax.set_ylim([0, 105]); ax.grid(True, alpha=0.3, axis="y")
for b, v in zip(bars2, mvals):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v*100:.1f}%",
            ha="center", va="bottom", fontweight="bold", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.subplots_adjust(hspace=0.65, wspace=0.40)
plt.savefig(path("multiclass_hybrid_output.png"), dpi=150, bbox_inches="tight")
if SHOW_PLOT:
    plt.show()
print("\n  [PLOT] Saved: multiclass_hybrid_output.png")

# =========================================
# STEP 8: EXPORT RESULTS
# =========================================
pd.DataFrame({
    "Model": ["SVM", "Random Forest", "Ensemble"],
    "Split": [split_mode] * 3,
    "Accuracy": [round(svm_accuracy, 4), round(rf_accuracy, 4), round(ens_accuracy, 4)],
    "Precision_macro": [round(precision_score(y_test, svm_pred, average="macro"), 4),
                        round(precision_score(y_test, rf_pred, average="macro"), 4),
                        round(ens_precision, 4)],
    "Recall_macro": [round(recall_score(y_test, svm_pred, average="macro"), 4),
                     round(recall_score(y_test, rf_pred, average="macro"), 4),
                     round(ens_recall, 4)],
    "F1_macro": [round(f1_score(y_test, svm_pred, average="macro"), 4),
                 round(f1_score(y_test, rf_pred, average="macro"), 4),
                 round(ens_f1, 4)],
}).assign(Coverage=[1.0, 1.0, 1.0]).pipe(lambda d: pd.concat([d, pd.DataFrame([{
    "Model": f"Ensemble after reject (th={REJECT_THRESHOLD})", "Split": split_mode,
    "Accuracy": round(sel_acc, 4), "Precision_macro": round(sel_prec, 4),
    "Recall_macro": round(sel_rec, 4), "F1_macro": round(sel_f1, 4),
    "Coverage": round(coverage, 4)}])], ignore_index=True)
).assign(CV_all_classes_in_every_fold=cv_all_classes).to_csv(path("multiclass_hybrid_results.csv"), index=False)
print("  [CSV]  Saved: multiclass_hybrid_results.csv")

# =========================================
# STEP 9: SAVE COMPLETE DEPLOYMENT BUNDLE
# =========================================
bundle = {
    "svm_pipeline": svm_pipe,            # scaler + SVC
    "rf_pipeline": rf_pipe,              # scaler + RandomForest
    "feature_cols": FEATURE_COLS,        # exact column order
    "label_names": LABEL_NAMES,
    "class_order": CLASS_ORDER,
    "malicious_labels": sorted(MALICIOUS_LABELS),
    "ensemble_rule": "mean of SVM and RF predict_proba, argmax over class_order",
    "reject_threshold": REJECT_THRESHOLD,
    "split_mode": split_mode,
    "snr_range": (float(df["SNR_dB"].min()), float(df["SNR_dB"].max())) if "SNR_dB" in df else None,
    "jsr_range": (float(df["JSR_dB"].min()), float(df["JSR_dB"].max())) if "JSR_dB" in df else None,
}
joblib.dump(bundle, path("ensemble_bundle_hybrid.pkl"))
# legacy files for result.py / classify_live.py
joblib.dump(svm_pipe.named_steps["svc"], path("svm_model_hybrid.pkl"))
joblib.dump(rf_model, path("rf_model_hybrid.pkl"))
joblib.dump(scaler_fitted, path("feature_scaler_hybrid.pkl"))
with open(path("ensemble_config_hybrid.json"), "w") as f:
    json.dump({k: v for k, v in bundle.items()
               if k not in ("svm_pipeline", "rf_pipeline")}, f, indent=2, default=str)

print("  [SAVE] ensemble_bundle_hybrid.pkl (+ svm_model_hybrid.pkl, rf_model_hybrid.pkl, feature_scaler_hybrid.pkl,"
      " ensemble_config_hybrid.json)")
print("""
  To classify a new feature vector x (shape [1, 13], columns = feature_cols):
      b = joblib.load('ensemble_bundle_hybrid.pkl')
      p = (b['svm_pipeline'].predict_proba(x) + b['rf_pipeline'].predict_proba(x)) / 2
      label = p.argmax(1)[0] if p.max() >= b['reject_threshold'] else 'Uncertain'
""")
print("=" * 60)
print("  Multi-Class Classification Complete")
print("=" * 60)