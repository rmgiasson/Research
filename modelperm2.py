import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
import lingam
import warnings
import sys
import os
warnings.filterwarnings("ignore")

# --------------------------------------------------
# Each SLURM array task is one permutation
# perm_idx=0 is the real data (no shuffling)
# perm_idx=1..N are shuffled permutations
# --------------------------------------------------
perm_idx    = int(sys.argv[1])
random_seed = 100 + perm_idx  # unique seed per permutation

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
# DAG: bootstrap only (best performing method)
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

# --------------------------------------------------
# Causal residuals + normalization
# --------------------------------------------------
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
# ElasticNet evaluation
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
# Load data
# --------------------------------------------------
df_t1 = pd.read_csv("T1_synthseg_vols_robust_no_parc.csv")
crs   = pd.read_csv("CRS_labels.csv").dropna()
df_t1["subject_id"] = df_t1["subject"].str.extract(r"(\d+)").astype(int)
df    = df_t1.merge(crs, left_on="subject_id", right_on="ID")

y_real      = df["neuropsych_score"].values
df_features = df[["total intracranial"] + brain_cols].copy()

# --------------------------------------------------
# Permute labels if this is not the real run
# perm_idx=0: real data
# perm_idx>0: shuffled labels
# --------------------------------------------------
if perm_idx == 0:
    y = y_real.copy()
    label = "real"
else:
    rng = np.random.RandomState(random_seed)
    y   = rng.permutation(y_real)
    label = "permuted"

print(f"Permutation {perm_idx} ({label}) | seed={random_seed}")

# --------------------------------------------------
# 5-fold CV — only run original_only and bootstrap_original_residual
# These are the two conditions needed for the delta comparison
# --------------------------------------------------
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)  # fixed across all permutations
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)  # isolates label shuffle variability only

fold_rows = []

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(df_features)):
    df_train = df_features.iloc[train_idx].reset_index(drop=True)
    df_test  = df_features.iloc[test_idx].reset_index(drop=True)
    y_train  = y[train_idx]
    y_test   = y[test_idx]

    X_train, X_test = preprocess_train_test(df_train, df_test)

    # original only
    mae, r2, r = evaluate_elasticnet(X_train, X_test, y_train, y_test, inner_cv)
    fold_rows.append({
        "perm_idx": perm_idx, "fold": fold_idx, "label": label,
        "condition": "original_only",
        "MAE": mae, "R2": r2, "pearson_r": r
    })

    # bootstrap residual only (best performing condition from repeated CV)
    adj = fit_dag_bootstrap(X_train, n_bootstrap=100, threshold=0.5,
                            random_state=random_seed)
    R_tr, R_te = normalize_residuals(
        compute_causal_residuals(X_train, adj),
        compute_causal_residuals(X_test,  adj)
    )
    mae, r2, r = evaluate_elasticnet(
        R_tr, R_te,
        y_train, y_test, inner_cv
    )
    fold_rows.append({
        "perm_idx": perm_idx, "fold": fold_idx, "label": label,
        "condition": "bootstrap_residual_only",
        "MAE": mae, "R2": r2, "pearson_r": r
    })

# Save results
out_df = pd.DataFrame(fold_rows)
os.makedirs("perm_results", exist_ok=True)
out_df.to_csv(f"perm_results/perm_{perm_idx:03d}.csv", index=False)
print(f"  Saved: perm_results/perm_{perm_idx:03d}.csv")

# Print fold means for monitoring
for cond in ["original_only", "bootstrap_residual_only"]:
    sub = out_df[out_df["condition"] == cond]
    print(f"  {cond}: MAE={sub['MAE'].mean():.4f} R²={sub['R2'].mean():.4f} r={sub['pearson_r'].nanmean():.4f}")
