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
    "left cerebral white matter",
    "left cerebral cortex",
    "left lateral ventricle",
    "left inferior lateral ventricle",
    "left cerebellum white matter",
    "left cerebellum cortex",
    "left thalamus",
    "left caudate",
    "left putamen",
    "left pallidum",
    "3rd ventricle",
    "4th ventricle",
    "brain-stem",
    "left hippocampus",
    "left amygdala",
    "left accumbens area",
    "left ventral DC",
    "right cerebral white matter",
    "right cerebral cortex",
    "right lateral ventricle",
    "right inferior lateral ventricle",
    "right cerebellum white matter",
    "right cerebellum cortex",
    "right thalamus",
    "right caudate",
    "right putamen",
    "right pallidum",
    "right hippocampus",
    "right amygdala",
    "right accumbens area",
    "right ventral DC"
]


# --------------------------------------------------
# Preprocessing
# ICV residualization is fit on HCP ONLY.
#
# The StandardScaler is still fit on the T1 training
# fold so the classifier preprocessing remains the
# same as the original pipeline.
#
# HCP is transformed using that same T1 scaler so
# HCP and T1 are on the same feature scale for the
# causal residual calculation.
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

        residuals[:, i] = (
            df[col].values
            - (slope * icv + intercept)
        )

    return residuals


def preprocess_train_test_hcp(df_train, df_test, hcp_df):
    """
    HCP is the ONLY population used to estimate the
    ICV residualization coefficients.

    T1 train/test are residualized using the HCP-derived
    coefficients.

    The StandardScaler is fit on T1 training data only,
    preserving the original classifier preprocessing.

    HCP is transformed using the same T1-training scaler
    so the HCP DAG and T1 causal residuals use the same
    feature scale.
    """

    # --------------------------------------------------
    # Fit ICV residualizer on HCP ONLY
    # --------------------------------------------------
    icv_coefs = fit_icv_residualizer(hcp_df)

    # --------------------------------------------------
    # Residualize T1 using HCP-derived coefficients
    # --------------------------------------------------
    resid_train = apply_icv_residualize(
        df_train,
        icv_coefs
    )

    resid_test = apply_icv_residualize(
        df_test,
        icv_coefs
    )

    # --------------------------------------------------
    # Residualize HCP using HCP-derived coefficients
    # --------------------------------------------------
    resid_hcp = apply_icv_residualize(
        hcp_df,
        icv_coefs
    )

    # --------------------------------------------------
    # Fit scaler on T1 training data ONLY
    # --------------------------------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(resid_train)
    X_test = scaler.transform(resid_test)

    # HCP gets the same scaler so that the DAG
    # coefficients are compatible with T1 features
    X_hcp = scaler.transform(resid_hcp)

    return X_train, X_test, X_hcp


# --------------------------------------------------
# Bootstrap DAG
# HCP ONLY
# --------------------------------------------------
def fit_dag_bootstrap(X_hcp, n_bootstrap=100, threshold=0.5,
                      random_state=42):
    """
    Fit a consensus DirectLiNGAM DAG using HCP ONLY.

    Each bootstrap iteration samples HCP subjects with
    replacement, fits DirectLiNGAM, and accumulates:

        1. Edge weights
        2. Edge occurrence counts

    An edge is retained if it appears in at least
    `threshold` fraction of successful bootstrap runs.

    No T1 or PPMI subjects are used in DAG construction.
    """

    rng = np.random.RandomState(random_state)

    n_samples, n_features = X_hcp.shape

    weight_sums = np.zeros(
        (n_features, n_features)
    )

    edge_counts = np.zeros(
        (n_features, n_features)
    )

    successful_runs = 0

    for b in range(n_bootstrap):

        # Bootstrap HCP subjects only
        idx = rng.choice(
            n_samples,
            size=n_samples,
            replace=True
        )

        X_boot = X_hcp[idx]

        try:
            model = lingam.DirectLiNGAM(
                random_state=b
            )

            model.fit(X_boot)

            adj = model.adjacency_matrix_

            # Accumulate edge weights
            weight_sums += adj

            # Count whether an edge was present
            edge_counts += (
                np.abs(adj) > 0
            ).astype(float)

            successful_runs += 1

        except Exception:
            continue

    if successful_runs == 0:
        raise RuntimeError(
            "All DirectLiNGAM bootstrap runs failed."
        )

    # Mean edge weight conditional on edge presence
    mean_weights = np.divide(
        weight_sums,
        edge_counts,
        out=np.zeros_like(weight_sums),
        where=edge_counts > 0
    )

    # Probability that each edge appears
    edge_probs = edge_counts / successful_runs

    # Keep only stable edges
    adj_consensus = np.where(
        edge_probs >= threshold,
        mean_weights,
        0.0
    )

    return adj_consensus


# --------------------------------------------------
# Compute causal residuals
# --------------------------------------------------
def compute_causal_residuals(X, adj):

    residuals = np.zeros_like(X)

    for i in range(X.shape[1]):

        # adj[i, j] != 0 means j -> i
        parents = np.where(
            adj[i, :] != 0
        )[0]

        if len(parents) == 0:
            residuals[:, i] = X[:, i]

        else:
            residuals[:, i] = (
                X[:, i]
                - X[:, parents] @ adj[i, parents]
            )

    return residuals


# --------------------------------------------------
# Normalize causal residuals
# --------------------------------------------------
def normalize_residuals(R_train, R_test):

    sigma = R_train.std(
        axis=0,
        ddof=1
    )

    sigma = np.where(
        sigma < 1e-8,
        1.0,
        sigma
    )

    return (
        R_train / sigma,
        R_test / sigma
    )


# --------------------------------------------------
# ElasticNet Logistic Regression evaluation
# class_weight="balanced" handles class imbalance
# Inner CV optimizes AUC
# --------------------------------------------------
param_grid = {
    "C":        [0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.1, 0.5, 0.9]
}


def evaluate_classifier(
    X_train,
    X_test,
    y_train,
    y_test,
    inner_cv
):

    gs = GridSearchCV(
        LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=5000,
            random_state=42
        ),
        param_grid,
        cv=inner_cv,
        scoring="roc_auc",
        refit=True,
        n_jobs=1
    )

    gs.fit(
        X_train,
        y_train
    )

    bp = gs.best_params_

    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=bp["C"],
        l1_ratio=bp["l1_ratio"],
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    bal_acc = balanced_accuracy_score(
        y_test,
        y_pred
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

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
#
# T1 is the classification/testing dataset because
# it contains the neuropsych scores.
# --------------------------------------------------
df_t1 = pd.read_csv(
    "T1_synthseg_vols_robust_no_parc.csv"
)

crs = pd.read_csv(
    "CRS_labels.csv"
)

crs = crs.dropna(
    subset=["neuropsych_score"]
)

df_t1["subject_id"] = (
    df_t1["subject"]
    .str.extract(r"(\d+)")
    .astype(int)
)

df = df_t1.merge(
    crs,
    left_on="subject_id",
    right_on="ID"
)


# --------------------------------------------------
# Define binary labels
#
# GO   = 0 (score < 3)
# NO-GO = 1 (score >= 3)
# --------------------------------------------------
df["label"] = (
    df["neuropsych_score"] >= 3
).astype(int)


# --------------------------------------------------
# HCP ONLY for DAG construction
#
# PPMI is NOT used.
# T1 is NOT used to construct the DAG.
# --------------------------------------------------
hcp_df = pd.read_csv(
    "synthseg_HCP.csv"
)

hcp_df = hcp_df[
    ["total intracranial"] + brain_cols
].copy()

hcp_df = hcp_df.dropna()


# --------------------------------------------------
# Print dataset information
# --------------------------------------------------
if repeat_idx == 0:

    print(
        f"Subjects after merge: {len(df)}"
    )

    print(
        f"GO  (score<3):  "
        f"{(df['label'] == 0).sum()}"
    )

    print(
        f"NO-GO (score>=3): "
        f"{(df['label'] == 1).sum()}"
    )

    print(
        f"HCP subjects used for DAG construction ONLY: "
        f"{len(hcp_df)}"
    )


# --------------------------------------------------
# T1 classification data
# --------------------------------------------------
y = df["label"].values

df_features = df[
    ["total intracranial"] + brain_cols
].copy()


# --------------------------------------------------
# 5-fold CV
# --------------------------------------------------
outer_cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=random_seed
)

inner_cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=random_seed
)


fold_rows = []


# --------------------------------------------------
# Outer CV
# --------------------------------------------------
for fold_idx, (train_idx, test_idx) in enumerate(
    outer_cv.split(df_features)
):

    # --------------------------------------------------
    # T1 train/test split
    # --------------------------------------------------
    df_train = (
        df_features
        .iloc[train_idx]
        .reset_index(drop=True)
    )

    df_test = (
        df_features
        .iloc[test_idx]
        .reset_index(drop=True)
    )

    y_train = y[train_idx]
    y_test = y[test_idx]


    # --------------------------------------------------
    # Preprocess
    #
    # ICV residualizer:
    #       HCP ONLY
    #
    # StandardScaler:
    #       T1 training fold ONLY
    # --------------------------------------------------
    X_train, X_test, X_hcp = (
        preprocess_train_test_hcp(
            df_train,
            df_test,
            hcp_df
        )
    )


    # --------------------------------------------------
    # Fit DAG using HCP ONLY
    #
    # No T1 subjects.
    # No PPMI subjects.
    # --------------------------------------------------
    adj = fit_dag_bootstrap(
        X_hcp,
        n_bootstrap=100,
        threshold=0.5,
        random_state=random_seed
    )


    # --------------------------------------------------
    # Causal residuals computed only for T1
    # --------------------------------------------------
    R_tr = compute_causal_residuals(
        X_train,
        adj
    )

    R_te = compute_causal_residuals(
        X_test,
        adj
    )

    Z_tr, Z_te = normalize_residuals(
        R_tr,
        R_te
    )


    # --------------------------------------------------
    # Condition 1: original only
    # --------------------------------------------------
    res = evaluate_classifier(
        X_train,
        X_test,
        y_train,
        y_test,
        inner_cv
    )

    res.update({
        "repeat": repeat_idx,
        "fold": fold_idx,
        "condition": "original_only"
    })

    fold_rows.append(res)


    # --------------------------------------------------
    # Condition 2: residual only
    # --------------------------------------------------
    res = evaluate_classifier(
        Z_tr,
        Z_te,
        y_train,
        y_test,
        inner_cv
    )

    res.update({
        "repeat": repeat_idx,
        "fold": fold_idx,
        "condition": "residual_only"
    })

    fold_rows.append(res)


    # --------------------------------------------------
    # Condition 3: original + residual
    # --------------------------------------------------
    res = evaluate_classifier(
        np.hstack([
            X_train,
            Z_tr
        ]),
        np.hstack([
            X_test,
            Z_te
        ]),
        y_train,
        y_test,
        inner_cv
    )

    res.update({
        "repeat": repeat_idx,
        "fold": fold_idx,
        "condition": "original_residual"
    })

    fold_rows.append(res)


# --------------------------------------------------
# Save results
# --------------------------------------------------
out_df = pd.DataFrame(
    fold_rows
)

os.makedirs(
    "classification_results_ppmi_hcp_dag",
    exist_ok=True
)

out_df.to_csv(
    f"classification_results_ppmi_hcp_dag/"
    f"repeat_{repeat_idx:02d}.csv",
    index=False
)


print(
    f"Repeat {repeat_idx} done."
)


for cond in [
    "original_only",
    "residual_only",
    "original_residual"
]:

    sub = out_df[
        out_df["condition"] == cond
    ]

    print(
        f"  {cond}: "
        f"bal_acc={sub['balanced_acc'].mean():.4f}  "
        f"auc={sub['auc'].mean():.4f}  "
        f"sens={sub['sensitivity'].mean():.4f}  "
        f"spec={sub['specificity'].mean():.4f}"
    )
