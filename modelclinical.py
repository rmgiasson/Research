import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, t as t_dist
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
clinical_cols = ["age", "sex", "education"]

# --------------------------------------------------
# Preprocessing: ICV residualize + standardize brain volumes
# Fit on train only
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
    """
    Returns:
      X_brain_train/test  — ICV-residualized + standardized volumes (31)
      X_clin_train/test   — standardized clinical features (3)
    Clinical features standardized separately so scale is comparable to brain features.
    """
    # Brain volumes
    icv_coefs   = fit_icv_residualizer(df_train)
    resid_train = apply_icv_residualize(df_train, icv_coefs)
    resid_test  = apply_icv_residualize(df_test,  icv_coefs)
    brain_scaler = StandardScaler()
    X_brain_train = brain_scaler.fit_transform(resid_train)
    X_brain_test  = brain_scaler.transform(resid_test)

    # Clinical features — standardize on train only
    clin_scaler = StandardScaler()
    X_clin_train = clin_scaler.fit_transform(
        df_train[clinical_cols].values.astype(float)
    )
    X_clin_test = clin_scaler.transform(
        df_test[clinical_cols].values.astype(float)
    )

    return X_brain_train, X_brain_test, X_clin_train, X_clin_test

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
    sigma = R_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return R_train / sigma, R_test / sigma

# --------------------------------------------------
# ElasticNet
# --------------------------------------------------
param_grid = {
    "alpha":    [0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.1, 0.5, 0.9]
}

def evaluate_elasticnet(X_train, X_test, y_train, y_test, inner_cv):
    gs = GridSearchCV(
        ElasticNet(max_iter=10000), param_grid,
        cv=inner_cv, scoring="neg_mean_absolute_error", refit=True
    )
    gs.fit(X_train, y_train)
    bp = gs.best_params_
    model = ElasticNet(alpha=bp["alpha"], l1_ratio=bp["l1_ratio"], max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    corr, _ = (np.nan, np.nan) if np.std(y_pred) < 1e-8 else pearsonr(y_test, y_pred)
    return mae, r2, corr

# --------------------------------------------------
# Load data — only keep subjects with complete data
# --------------------------------------------------
df_t1 = pd.read_csv("T1_synthseg_vols_robust_no_parc.csv")
crs   = pd.read_csv("CRS_labels_new.csv").dropna()

df_t1["subject_id"] = df_t1["subject"].str.extract(r"(\d+)").astype(int)
df = df_t1.merge(crs, left_on="subject_id", right_on="ID")

# Drop subjects missing any clinical or target variable
required_cols = clinical_cols + ["neuropsych_score"]
before = len(df)
df = df.dropna(subset=required_cols)
after  = len(df)
if repeat_idx == 0:
    print(f"Subjects after merge: {before}")
    print(f"Subjects after dropping missing clinical/target: {after}")
    print(f"Dropped: {before - after}")

y           = df["neuropsych_score"].values
df_brain    = df[["total intracranial"] + brain_cols].copy()
df_clinical = df[clinical_cols].copy()

# --------------------------------------------------
# 5-fold CV
# --------------------------------------------------
outer_cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)

fold_rows = []

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(df_brain)):
    df_brain_train = df_brain.iloc[train_idx].reset_index(drop=True)
    df_brain_test  = df_brain.iloc[test_idx].reset_index(drop=True)
    df_clin_train  = df_clinical.iloc[train_idx].reset_index(drop=True)
    df_clin_test   = df_clinical.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test  = y[test_idx]

    X_brain_train, X_brain_test, X_clin_train, X_clin_test = preprocess_train_test(
        pd.concat([df_brain_train, df_clin_train], axis=1),
        pd.concat([df_brain_test,  df_clin_test],  axis=1)
    )

    # Fit DAG once per fold on brain volumes only
    adj  = fit_dag_bootstrap(X_brain_train, n_bootstrap=100,
                             threshold=0.5, random_state=random_seed)
    R_tr = compute_causal_residuals(X_brain_train, adj)
    R_te = compute_causal_residuals(X_brain_test,  adj)
    Z_tr, Z_te = normalize_residuals(R_tr, R_te)

    # --- Condition 1: clinical only ---
    mae, r2, r = evaluate_elasticnet(
        X_clin_train, X_clin_test, y_train, y_test, inner_cv
    )
    fold_rows.append({"repeat": repeat_idx, "fold": fold_idx,
                      "condition": "clinical_only",
                      "MAE": mae, "R2": r2, "pearson_r": r})

    # --- Condition 2: clinical + original volumes ---
    mae, r2, r = evaluate_elasticnet(
        np.hstack([X_clin_train, X_brain_train]),
        np.hstack([X_clin_test,  X_brain_test]),
        y_train, y_test, inner_cv
    )
    fold_rows.append({"repeat": repeat_idx, "fold": fold_idx,
                      "condition": "clinical_original",
                      "MAE": mae, "R2": r2, "pearson_r": r})

    # --- Condition 3: clinical + residuals ---
    mae, r2, r = evaluate_elasticnet(
        np.hstack([X_clin_train, Z_tr]),
        np.hstack([X_clin_test,  Z_te]),
        y_train, y_test, inner_cv
    )
    fold_rows.append({"repeat": repeat_idx, "fold": fold_idx,
                      "condition": "clinical_residual",
                      "MAE": mae, "R2": r2, "pearson_r": r})

# Save
out_df = pd.DataFrame(fold_rows)
os.makedirs("clinical_results", exist_ok=True)
out_df.to_csv(f"clinical_results/repeat_{repeat_idx:02d}.csv", index=False)
print(f"Repeat {repeat_idx} done.")
for cond in ["clinical_only", "clinical_original", "clinical_residual"]:
    sub = out_df[out_df["condition"] == cond]
    print(f"  {cond}: MAE={sub['MAE'].mean():.4f}  "
          f"R²={sub['R2'].mean():.4f}  "
          f"r={sub['pearson_r'].mean():.4f}")
