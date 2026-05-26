import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
import lingam
import warnings
warnings.filterwarnings("ignore")

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

thresholds = [0.3, 0.5, 0.7, 0.9]

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
# Bootstrap DAG — single fit, returns both mean_weights and edge_probs
# Threshold applied separately so bootstrap only runs once per fold
# --------------------------------------------------
def fit_dag_bootstrap(X_train, n_bootstrap=100, random_state=42):
    """
    Fit bootstrap DAG once. Returns mean_weights and edge_probs separately
    so multiple thresholds can be applied without refitting.
    adj[i,j] means j -> i.
    """
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
    print(f"  Bootstrap complete: {successful_runs}/{n_bootstrap} successful runs")
    return mean_weights, edge_probs

def apply_threshold(mean_weights, edge_probs, threshold):
    """Apply threshold to pre-computed bootstrap results."""
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

def evaluate_elasticnet(X_train, X_test, y_train, y_test, inner_cv,
                        feature_names=None):
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
    return mae, r2, corr, model.coef_

# --------------------------------------------------
# Load data
# --------------------------------------------------
df_t1 = pd.read_csv("T1_synthseg_vols_robust_no_parc.csv")
crs   = pd.read_csv("CRS_labels.csv").dropna()
df_t1["subject_id"] = df_t1["subject"].str.extract(r"(\d+)").astype(int)
df    = df_t1.merge(crs, left_on="subject_id", right_on="ID")
print(f"Subjects after merge: {len(df)}")

y           = df["neuropsych_score"].values
df_features = df[["total intracranial"] + brain_cols].copy()

resid_feature_names = [f"resid_{c}" for c in brain_cols]

# --------------------------------------------------
# Nested CV
# --------------------------------------------------
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

perf_rows  = []
coef_store = {t: [] for t in thresholds}

# --------------------------------------------------
# Outer CV loop
# --------------------------------------------------
for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(df_features)):
    print(f"\n--- Outer Fold {fold_idx + 1} ---")

    df_train = df_features.iloc[train_idx].reset_index(drop=True)
    df_test  = df_features.iloc[test_idx].reset_index(drop=True)
    y_train  = y[train_idx]
    y_test   = y[test_idx]

    X_train, X_test = preprocess_train_test(df_train, df_test)

    # Fit bootstrap DAG exactly once per outer fold
    print("  Fitting bootstrap DAG (100 runs)...")
    mean_weights, edge_probs = fit_dag_bootstrap(X_train, n_bootstrap=100,
                                                  random_state=42)

    # Apply each threshold to the same bootstrap result
    for threshold in thresholds:
        adj      = apply_threshold(mean_weights, edge_probs, threshold)
        n_edges  = int((adj != 0).sum())

        # Residual only
        R_tr = compute_causal_residuals(X_train, adj)
        R_te = compute_causal_residuals(X_test,  adj)
        Z_tr, Z_te = normalize_residuals(R_tr, R_te)

        mae, r2, corr, coef = evaluate_elasticnet(
            Z_tr, Z_te, y_train, y_test, inner_cv,
            feature_names=resid_feature_names
        )

        print(f"  Threshold={threshold}: n_edges={n_edges}  "
              f"MAE={mae:.4f}  R²={r2:.4f}  r={corr:.4f}")

        perf_rows.append({
            "threshold": threshold,
            "fold":      fold_idx + 1,
            "n_edges":   n_edges,
            "MAE":       mae,
            "R2":        r2,
            "pearson_r": corr
        })
        coef_store[threshold].append(coef)

# --------------------------------------------------
# Performance summary
# --------------------------------------------------
print(f"\n{'='*60}")
print("PERFORMANCE SUMMARY BY THRESHOLD (residual only)")
print(f"{'='*60}")

perf_df = pd.DataFrame(perf_rows)
summary_rows = []
for threshold in thresholds:
    sub = perf_df[perf_df["threshold"] == threshold]
    summary_rows.append({
        "threshold":      threshold,
        "mean_n_edges":   sub["n_edges"].mean(),
        "mean_MAE":       sub["MAE"].mean(),
        "std_MAE":        sub["MAE"].std(),
        "mean_R2":        sub["R2"].mean(),
        "std_R2":         sub["R2"].std(),
        "mean_pearson_r": sub["pearson_r"].mean(),
        "std_pearson_r":  sub["pearson_r"].std(),
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# --------------------------------------------------
# Residual feature stability per threshold
# --------------------------------------------------
print(f"\n{'='*60}")
print("RESIDUAL FEATURE STABILITY BY THRESHOLD")
print(f"{'='*60}")

stability_rows = []
for threshold in thresholds:
    coef_matrix = np.array(coef_store[threshold])
    n_folds     = coef_matrix.shape[0]

    sel_freq      = (coef_matrix != 0).mean(axis=0)
    majority_sign = np.sign(coef_matrix.sum(axis=0))
    signs         = np.sign(coef_matrix)
    sign_cons     = np.where(
        (coef_matrix != 0).sum(axis=0) > 0,
        (signs == majority_sign).sum(axis=0) / n_folds,
        np.nan
    )
    mean_coef     = coef_matrix.mean(axis=0)

    n_active = int((sel_freq > 0).sum())
    n_stable = int((sel_freq == 1.0).sum())

    print(f"\n  Threshold={threshold}  "
          f"(mean_edges={summary_df[summary_df['threshold']==threshold]['mean_n_edges'].values[0]:.1f}):")
    print(f"  Active residual features: {n_active}/{len(brain_cols)}")
    print(f"  Stable across all folds:  {n_stable}/{len(brain_cols)}")
    print(f"\n  {'Feature':<45} {'Sel.Freq':>9} {'Sign.Cons':>10} {'Mean.Coef':>10}")

    for name, sf, sc, mc in sorted(
        zip(resid_feature_names, sel_freq, sign_cons, mean_coef),
        key=lambda x: -x[1]
    ):
        if sf > 0:
            print(f"  {name:<45} {sf:>9.2f} {sc:>10.2f} {mc:>10.4f}")

    stability_rows.append({
        "threshold":      threshold,
        "mean_n_edges":   summary_df[summary_df["threshold"]==threshold]["mean_n_edges"].values[0],
        "n_active":       n_active,
        "n_stable":       n_stable,
        "top_feature":    resid_feature_names[np.argmax(sel_freq)],
        "top_sel_freq":   sel_freq.max()
    })

stability_df = pd.DataFrame(stability_rows)

# --------------------------------------------------
# Save
# --------------------------------------------------
perf_df.to_csv("threshold_sweep_performance.csv", index=False)
summary_df.to_csv("threshold_sweep_summary.csv", index=False)
stability_df.to_csv("threshold_sweep_stability.csv", index=False)

print(f"\n{'='*60}")
print("STABILITY SUMMARY ACROSS THRESHOLDS")
print(f"{'='*60}")
print(stability_df.to_string(index=False))

print("\nSaved:")
print("  threshold_sweep_performance.csv")
print("  threshold_sweep_summary.csv")
print("  threshold_sweep_stability.csv")
