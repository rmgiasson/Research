import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             confusion_matrix)
import lingam
import warnings
import sys
import os
warnings.filterwarnings("ignore")

repeat_idx  = int(sys.argv[1])
random_seed = 42 + repeat_idx

# --------------------------------------------------
# Brain volume columns
# --------------------------------------------------
brain_cols = [
    "left cerebral white matter", "left cerebral cortex",
    "left lateral ventricle", "left inferior lateral ventricle",
    "left cerebellum white matter", "left cerebellum cortex",
    "left thalamus", "left caudate", "left putamen", "left pallidum",
    "3rd ventricle", "4th ventricle", "brain-stem",
    "left hippocampus", "left amygdala", "left accumbens area",
    "left ventral DC", "right cerebral white matter",
    "right cerebral cortex", "right lateral ventricle",
    "right inferior lateral ventricle", "right cerebellum white matter",
    "right cerebellum cortex", "right thalamus", "right caudate",
    "right putamen", "right pallidum", "right hippocampus",
    "right amygdala", "right accumbens area", "right ventral DC"
]

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
def fit_icv_residualizer(df_train):
    icv = df_train["total intracranial"].values.reshape(-1, 1)
    coefs = {}
    for col in brain_cols:
        reg = LinearRegression()
        reg.fit(icv, df_train[col].values)
        coefs[col] = (reg.coef_[0], reg.intercept_)
    return coefs

def apply_icv_residualize(df, coefs):
    icv = df["total intracranial"].values.flatten()
    residuals = np.zeros((len(df), len(brain_cols)))
    for i, col in enumerate(brain_cols):
        slope, intercept = coefs[col]
        residuals[:, i] = df[col].values - (slope * icv + intercept)
    return residuals

def preprocess_train_test(df_train, df_test):
    icv_coefs = fit_icv_residualizer(df_train)
    resid_train = apply_icv_residualize(df_train, icv_coefs)
    resid_test  = apply_icv_residualize(df_test,  icv_coefs)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(resid_train)
    X_test  = scaler.transform(resid_test)
    return X_train, X_test

# --------------------------------------------------
# Bootstrap DAG
# --------------------------------------------------
def fit_dag_bootstrap(X_train, n_bootstrap=100, threshold=0.5, random_state=42):
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X_train.shape
    weight_sums = np.zeros((n_features, n_features))
    edge_counts = np.zeros((n_features, n_features))
    successful_runs = 0
    for b in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        try:
            model = lingam.DirectLiNGAM(random_state=b)
            model.fit(X_train[idx])
            adj = model.adjacency_matrix_
            weight_sums += adj
            edge_counts += (np.abs(adj) > 0).astype(float)
            successful_runs += 1
        except Exception:
            continue
    mean_weights = np.divide(
        weight_sums, edge_counts,
        out=np.zeros_like(weight_sums), where=edge_counts > 0
    )
    edge_probs = edge_counts / max(successful_runs, 1)
    return np.where(edge_probs >= threshold, mean_weights, 0.0)

def compute_causal_residuals(X, adj):
    residuals = np.zeros_like(X)
    for i in range(X.shape[1]):
        parents = np.where(adj[i, :] != 0)[0]
        if len(parents) == 0:
            residuals[:, i] = X[:, i]
        else:
            residuals[:, i] = X[:, i] - X[:, parents] @ adj[i, parents]
    return residuals

def normalize_residuals(R_train, R_test):
    sigma = R_train.std(axis=0, ddof=1)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return R_train / sigma, R_test / sigma

# --------------------------------------------------
# ElasticNet Logistic Regression evaluation
# class_weight="balanced" handles class imbalance
# Inner CV optimizes AUC
# --------------------------------------------------
param_grid = {
    "C":        [0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.1, 0.5, 0.9]
}

def evaluate_classifier(X_train, X_test, y_train, y_test, inner_cv):
    gs = GridSearchCV(
        LogisticRegression(
            penalty="elasticnet", solver="saga",
            class_weight="balanced", max_iter=5000,
            random_state=42
        ),
        param_grid,
        cv=inner_cv,
        scoring="roc_auc",
        refit=True,
        n_jobs=1
    )
    gs.fit(X_train, y_train)
    bp = gs.best_params_

    model = LogisticRegression(
        penalty="elasticnet", solver="saga",
        C=bp["C"], l1_ratio=bp["l1_ratio"],
        class_weight="balanced", max_iter=5000,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    bal_acc  = balanced_accuracy_score(y_test, y_pred)
    auc      = roc_auc_score(y_test, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "balanced_acc": bal_acc,
        "auc":          auc,
        "sensitivity":  sensitivity,
        "specificity":  specificity,
        "best_C":       bp["C"],
        "best_l1ratio": bp["l1_ratio"]
    }

# --------------------------------------------------
# Load data
# --------------------------------------------------
df_t1 = pd.read_csv("T1_synthseg_vols_robust_no_parc.csv")
crs   = pd.read_csv("CRS_labels.csv")
crs   = crs.dropna(subset=["neuropsych_score"])
df_t1["subject_id"] = df_t1["subject"].str.extract(r"(\d+)").astype(int)
df    = df_t1.merge(crs, left_on="subject_id", right_on="ID")

# Define binary labels: GO=0 (score<3), NO-GO=1 (score>=3)
df["label"] = (df["neuropsych_score"] >= 3).astype(int)

if repeat_idx == 0:
    print(f"Subjects after merge: {len(df)}")
    print(f"GO  (score<3):  {(df['label']==0).sum()}")
    print(f"NO-GO (score>=3): {(df['label']==1).sum()}")

y           = df["label"].values
df_features = df[["total intracranial"] + brain_cols].copy()

# --------------------------------------------------
# 5-fold CV
# --------------------------------------------------
outer_cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)

fold_rows = []

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(df_features)):
    df_train = df_features.iloc[train_idx].reset_index(drop=True)
    df_test  = df_features.iloc[test_idx].reset_index(drop=True)
    y_train  = y[train_idx]
    y_test   = y[test_idx]

    X_train, X_test = preprocess_train_test(df_train, df_test)

    # Fit DAG once per fold
    adj  = fit_dag_bootstrap(X_train, n_bootstrap=100,
                             threshold=0.5, random_state=random_seed)
    R_tr = compute_causal_residuals(X_train, adj)
    R_te = compute_causal_residuals(X_test,  adj)
    Z_tr, Z_te = normalize_residuals(R_tr, R_te)

    # --- Condition 1: original only ---
    res = evaluate_classifier(X_train, X_test, y_train, y_test, inner_cv)
    res.update({"repeat": repeat_idx, "fold": fold_idx,
                "condition": "original_only"})
    fold_rows.append(res)

    # --- Condition 2: residual only ---
    res = evaluate_classifier(Z_tr, Z_te, y_train, y_test, inner_cv)
    res.update({"repeat": repeat_idx, "fold": fold_idx,
                "condition": "residual_only"})
    fold_rows.append(res)

    # --- Condition 3: original + residual ---
    res = evaluate_classifier(
        np.hstack([X_train, Z_tr]),
        np.hstack([X_test,  Z_te]),
        y_train, y_test, inner_cv
    )
    res.update({"repeat": repeat_idx, "fold": fold_idx,
                "condition": "original_residual"})
    fold_rows.append(res)

# Save
out_df = pd.DataFrame(fold_rows)
os.makedirs("classification_results", exist_ok=True)
out_df.to_csv(f"classification_results/repeat_{repeat_idx:02d}.csv", index=False)
print(f"Repeat {repeat_idx} done.")
for cond in ["original_only", "residual_only", "original_residual"]:
    sub = out_df[out_df["condition"] == cond]
    print(f"  {cond}: "
          f"bal_acc={sub['balanced_acc'].mean():.4f}  "
          f"auc={sub['auc'].mean():.4f}  "
          f"sens={sub['sensitivity'].mean():.4f}  "
          f"spec={sub['specificity'].mean():.4f}")
