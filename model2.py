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

# --------------------------------------------------
# Preprocessing: fit on train, apply to both
# --------------------------------------------------
def fit_icv_residualizer(df_train):
    """Fit ICV regression on training data only, return coefficients."""
    icv = df_train["total intracranial"].values.reshape(-1, 1)
    coefs = {}
    for col in brain_cols:
        reg = LinearRegression()
        reg.fit(icv, df_train[col].values)
        coefs[col] = (reg.coef_[0], reg.intercept_)
    return coefs

def apply_icv_residualize(df, coefs):
    """Apply pre-fit ICV regression to residualize a dataframe."""
    icv = df["total intracranial"].values.flatten()
    residuals = np.zeros((len(df), len(brain_cols)))
    for i, col in enumerate(brain_cols):
        slope, intercept = coefs[col]
        residuals[:, i] = df[col].values - (slope * icv + intercept)
    return residuals

def preprocess_train_test(df_train, df_test):
    """
    ICV residualize then standardize.
    Fit on train only, apply to both. Returns X_train, X_test, scaler.
    """
    icv_coefs = fit_icv_residualizer(df_train)
    resid_train = apply_icv_residualize(df_train, icv_coefs)
    resid_test  = apply_icv_residualize(df_test,  icv_coefs)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(resid_train)
    X_test  = scaler.transform(resid_test)

    return X_train, X_test, icv_coefs, scaler

# --------------------------------------------------
# DAG fitting
# adj[i,j] means j -> i (parent j causes child i)
# --------------------------------------------------
def fit_dag_single(X_train, random_state=42):
    """
    Fit a single DirectLiNGAM on training data.
    Returns adjacency matrix (n_features x n_features).
    """
    model = lingam.DirectLiNGAM(random_state=random_state)
    model.fit(X_train)
    return model.adjacency_matrix_

def fit_dag_bootstrap(X_train, n_bootstrap=100, threshold=0.5, random_state=42):
    """
    Fit DirectLiNGAM on n_bootstrap subsamples.
    Accumulates actual edge weights and edge presence counts separately.
    Consensus matrix uses mean causal weight for stable edges (freq >= threshold),
    zero for unstable edges. This preserves the biological meaning of A_ij as a
    causal coefficient, not an occurrence frequency.
    adj[i,j] means j -> i (parent j causes child i).
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X_train.shape
    weight_sums = np.zeros((n_features, n_features))
    edge_counts = np.zeros((n_features, n_features))
    successful_runs = 0

    for b in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_train[idx]
        try:
            model = lingam.DirectLiNGAM(random_state=b)
            model.fit(X_boot)
            adj = model.adjacency_matrix_
            # Accumulate actual weights and presence counts separately
            weight_sums += adj
            edge_counts += (np.abs(adj) > 0).astype(float)
            successful_runs += 1
        except Exception:
            continue

    print(f"  Successful bootstrap runs: {successful_runs}/{n_bootstrap}")

    # Mean causal weight across runs where edge was present
    mean_weights = np.divide(
        weight_sums,
        edge_counts,
        out=np.zeros_like(weight_sums),
        where=edge_counts > 0
    )

    # Edge stability: fraction of runs where edge appeared
    edge_probs = edge_counts / successful_runs

    # Consensus: keep mean weight only for stable edges, zero otherwise
    consensus_adj = np.where(edge_probs >= threshold, mean_weights, 0.0)

    print(f"  Bootstrap DAG: {int((consensus_adj != 0).sum())} edges retained "
          f"(stability threshold={threshold})")

    return consensus_adj

# --------------------------------------------------
# Compute causal residual features from adjacency matrix
# adj[i,j] means j -> i, so Pa(i) = columns j where adj[i,j] != 0
# R_i = x_i - sum_{j in Pa(i)} adj[i,j] * x_j
# --------------------------------------------------
def compute_causal_residuals(X, adj):
    """
    X:   (n_samples, n_features) standardized brain volumes
    adj: (n_features, n_features) adjacency matrix, adj[i,j] = weight j->i
    Returns residuals of same shape as X.
    """
    n_samples, n_features = X.shape
    residuals = np.zeros_like(X)
    for i in range(n_features):
        # Parent nodes of i: columns j where adj[i,j] != 0
        parents = np.where(adj[i, :] != 0)[0]
        if len(parents) == 0:
            # Root node: residual is just the original value
            residuals[:, i] = X[:, i]
        else:
            x_hat = X[:, parents] @ adj[i, parents]
            residuals[:, i] = X[:, i] - x_hat
    return residuals

# --------------------------------------------------
# Evaluate one condition: fit model, return metrics
# --------------------------------------------------
def evaluate_condition(X_train, X_test, y_train, y_test, param_grid, inner_cv):
    """Run inner CV to select hyperparams, train final model, return metrics."""
    inner_model = GridSearchCV(
        ElasticNet(max_iter=10000),
        param_grid,
        cv=inner_cv,
        scoring="neg_mean_absolute_error",
        refit=True
    )
    inner_model.fit(X_train, y_train)
    best_params = inner_model.best_params_

    final_model = ElasticNet(
        alpha=best_params["alpha"],
        l1_ratio=best_params["l1_ratio"],
        max_iter=10000
    )
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    # Guard against constant predictions (e.g. high alpha collapses to mean)
    if np.std(y_pred) < 1e-8:
        corr, pval = np.nan, np.nan
    else:
        corr, pval = pearsonr(y_test, y_pred)

    return {
        "best_params": best_params,
        "MAE":         mae,
        "R2":          r2,
        "pearson_r":   corr,
        "pearson_p":   pval
    }

# --------------------------------------------------
# Load and merge data
# --------------------------------------------------
df_t1 = pd.read_csv("/blue/neurology-dept/JOSH/_UF_DBS/T1_synthseg_vols_robust_no_parc.csv")
crs   = pd.read_csv("CRS_labels.csv").dropna()

df_t1["subject_id"] = df_t1["subject"].str.extract(r"(\d+)").astype(int)
df = df_t1.merge(crs, left_on="subject_id", right_on="ID")
print(f"Subjects after merge: {len(df)}")

y            = df["neuropsych_score"].values
df_features  = df[["total intracranial"] + brain_cols].copy()

# --------------------------------------------------
# Nested CV setup
# --------------------------------------------------
outer_cv = KFold(n_splits=2, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=2, shuffle=True, random_state=42)

param_grid = {
    "alpha":    [0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.1, 0.5, 0.9]
}

# Conditions to evaluate
conditions = [
    "original_only",           # baseline, no DAG
    "single_residual_only",    # single-run DAG, residuals only
    "single_original_residual",# single-run DAG, original + residuals
    "bootstrap_residual_only", # bootstrapped DAG, residuals only
    "bootstrap_original_residual" # bootstrapped DAG, original + residuals
]

# Store results per condition per fold
all_results = {c: [] for c in conditions}

# --------------------------------------------------
# Outer CV loop
# --------------------------------------------------
for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(df_features)):
    print(f"\n{'='*60}")
    print(f"OUTER FOLD {fold_idx + 1}")
    print(f"{'='*60}")

    df_train = df_features.iloc[train_idx].reset_index(drop=True)
    df_test  = df_features.iloc[test_idx].reset_index(drop=True)
    y_train  = y[train_idx]
    y_test   = y[test_idx]

    # Preprocess: ICV residualize + standardize (fit on train only)
    X_train, X_test, icv_coefs, scaler = preprocess_train_test(df_train, df_test)

    # --------------------------------------------------
    # Condition 1: Original only (baseline)
    # --------------------------------------------------
    print("\n[1/5] Original only (baseline)...")
    res = evaluate_condition(X_train, X_test, y_train, y_test, param_grid, inner_cv)
    res.update({"fold": fold_idx + 1, "n_train": len(y_train), "n_test": len(y_test)})
    all_results["original_only"].append(res)
    print(f"  MAE={res['MAE']:.4f}  R²={res['R2']:.4f}  r={res['pearson_r']:.4f}")

    # --------------------------------------------------
    # Fit single-run DAG on training data
    # --------------------------------------------------
    print("\n  Fitting single-run DirectLiNGAM...")
    adj_single = fit_dag_single(X_train)
    causal_resid_train_single = compute_causal_residuals(X_train, adj_single)
    causal_resid_test_single  = compute_causal_residuals(X_test,  adj_single)

    # --------------------------------------------------
    # Condition 2: Single DAG — residuals only
    # --------------------------------------------------
    print("\n[2/5] Single DAG — residuals only...")
    res = evaluate_condition(
        causal_resid_train_single, causal_resid_test_single,
        y_train, y_test, param_grid, inner_cv
    )
    res.update({"fold": fold_idx + 1, "n_train": len(y_train), "n_test": len(y_test)})
    all_results["single_residual_only"].append(res)
    print(f"  MAE={res['MAE']:.4f}  R²={res['R2']:.4f}  r={res['pearson_r']:.4f}")

    # --------------------------------------------------
    # Condition 3: Single DAG — original + residuals
    # --------------------------------------------------
    print("\n[3/5] Single DAG — original + residuals...")
    X_train_combined_single = np.hstack([X_train, causal_resid_train_single])
    X_test_combined_single  = np.hstack([X_test,  causal_resid_test_single])
    res = evaluate_condition(
        X_train_combined_single, X_test_combined_single,
        y_train, y_test, param_grid, inner_cv
    )
    res.update({"fold": fold_idx + 1, "n_train": len(y_train), "n_test": len(y_test)})
    all_results["single_original_residual"].append(res)
    print(f"  MAE={res['MAE']:.4f}  R²={res['R2']:.4f}  r={res['pearson_r']:.4f}")

    # --------------------------------------------------
    # Fit bootstrapped DAG on training data
    # --------------------------------------------------
    print("\n  Fitting bootstrapped DirectLiNGAM (100 runs)...")
    adj_boot = fit_dag_bootstrap(X_train, n_bootstrap=100, threshold=0.5)
    causal_resid_train_boot = compute_causal_residuals(X_train, adj_boot)
    causal_resid_test_boot  = compute_causal_residuals(X_test,  adj_boot)

    # --------------------------------------------------
    # Condition 4: Bootstrap DAG — residuals only
    # --------------------------------------------------
    print("\n[4/5] Bootstrap DAG — residuals only...")
    res = evaluate_condition(
        causal_resid_train_boot, causal_resid_test_boot,
        y_train, y_test, param_grid, inner_cv
    )
    res.update({"fold": fold_idx + 1, "n_train": len(y_train), "n_test": len(y_test)})
    all_results["bootstrap_residual_only"].append(res)
    print(f"  MAE={res['MAE']:.4f}  R²={res['R2']:.4f}  r={res['pearson_r']:.4f}")

    # --------------------------------------------------
    # Condition 5: Bootstrap DAG — original + residuals
    # --------------------------------------------------
    print("\n[5/5] Bootstrap DAG — original + residuals...")
    X_train_combined_boot = np.hstack([X_train, causal_resid_train_boot])
    X_test_combined_boot  = np.hstack([X_test,  causal_resid_test_boot])
    res = evaluate_condition(
        X_train_combined_boot, X_test_combined_boot,
        y_train, y_test, param_grid, inner_cv
    )
    res.update({"fold": fold_idx + 1, "n_train": len(y_train), "n_test": len(y_test)})
    all_results["bootstrap_original_residual"].append(res)
    print(f"  MAE={res['MAE']:.4f}  R²={res['R2']:.4f}  r={res['pearson_r']:.4f}")

# --------------------------------------------------
# Summary across all conditions
# --------------------------------------------------
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")

summary_rows = []
for condition, fold_results in all_results.items():
    maes = [r["MAE"] for r in fold_results]
    r2s  = [r["R2"]  for r in fold_results]
    rs   = [r["pearson_r"] for r in fold_results]
    summary_rows.append({
        "condition":      condition,
        "mean_MAE":       np.mean(maes),
        "std_MAE":        np.std(maes),
        "mean_R2":        np.mean(r2s),
        "std_R2":         np.std(r2s),
        "mean_pearson_r": np.mean(rs),
        "std_pearson_r":  np.std(rs),
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv("nested_cv_dag_results.csv", index=False)
print("\nSaved: nested_cv_dag_results.csv")
