import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.cross_decomposition import PLSRegression
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
    Fit on train only, apply to both.
    Returns X_train, X_test, icv_coefs, scaler.
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
    """Fit a single DirectLiNGAM on training data."""
    model = lingam.DirectLiNGAM(random_state=random_state)
    model.fit(X_train)
    return model.adjacency_matrix_

def fit_dag_bootstrap(X_train, n_bootstrap=100, threshold=0.5, random_state=42):
    """
    Fit DirectLiNGAM on n_bootstrap subsamples.
    Accumulates actual edge weights and presence counts separately.
    Consensus matrix uses mean causal weight for stable edges (freq >= threshold),
    zero for unstable edges. Preserves biological meaning of A_ij as causal
    coefficient, not occurrence frequency. adj[i,j] means j -> i.
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
# R_i = x_i - sum_{j in Pa(i)} A_ij * x_j
# Then normalize: Z_i = R_i / sigma_i
# sigma_i computed from train fold residuals only (no leakage)
# --------------------------------------------------
def compute_causal_residuals(X, adj):
    """
    Compute raw causal residuals R_i = x_i - x_i_hat.
    X:   (n_samples, n_features)
    adj: (n_features, n_features), adj[i,j] = causal weight j->i
    Returns raw residuals, same shape as X.
    """
    n_samples, n_features = X.shape
    residuals = np.zeros_like(X)
    for i in range(n_features):
        parents = np.where(adj[i, :] != 0)[0]
        if len(parents) == 0:
            residuals[:, i] = X[:, i]
        else:
            x_hat = X[:, parents] @ adj[i, parents]
            residuals[:, i] = X[:, i] - x_hat
    return residuals

def normalize_residuals(R_train, R_test):
    """
    Normalize causal residuals by per-node std computed from train fold only.
    Z_i = R_i / sigma_i  (sigma_i from train residuals)
    Avoids division by zero for root nodes with zero variance.
    """
    sigma = R_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0  # root nodes: std~0, leave unchanged
    Z_train = R_train / sigma
    Z_test  = R_test  / sigma
    return Z_train, Z_test

# --------------------------------------------------
# Evaluate one condition with ElasticNet
# Returns metrics + coefficients for inspection
# --------------------------------------------------
def evaluate_elasticnet(X_train, X_test, y_train, y_test,
                        param_grid, inner_cv, feature_names=None):
    """Inner CV selects hyperparams, final model trained on full outer train."""
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
    if np.std(y_pred) < 1e-8:
        corr, pval = np.nan, np.nan
    else:
        corr, pval = pearsonr(y_test, y_pred)

    result = {
        "best_params": best_params,
        "MAE":         mae,
        "R2":          r2,
        "pearson_r":   corr,
        "pearson_p":   pval,
        "coef":        final_model.coef_,
        "feature_names": feature_names if feature_names is not None else []
    }
    return result

# --------------------------------------------------
# Evaluate one condition with PLSRegression
# --------------------------------------------------
def evaluate_pls(X_train, X_test, y_train, y_test, inner_cv):
    """Inner CV selects n_components (2-10), final PLS trained on full outer train."""
    best_score = -np.inf
    best_n = 2

    for n in range(2, 11):
        # Cap n_components at min(n_samples, n_features) - 1 to avoid PLS errors
        max_comp = min(X_train.shape[0], X_train.shape[1]) - 1
        if n > max_comp:
            break
        fold_scores = []
        for tr_idx, val_idx in inner_cv.split(X_train):
            pls = PLSRegression(n_components=n)
            pls.fit(X_train[tr_idx], y_train[tr_idx])
            pred = pls.predict(X_train[val_idx]).flatten()
            fold_scores.append(-mean_absolute_error(y_train[val_idx], pred))
        score = np.mean(fold_scores)
        if score > best_score:
            best_score = score
            best_n = n

    final_pls = PLSRegression(n_components=best_n)
    final_pls.fit(X_train, y_train)
    y_pred = final_pls.predict(X_test).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    if np.std(y_pred) < 1e-8:
        corr, pval = np.nan, np.nan
    else:
        corr, pval = pearsonr(y_test, y_pred)

    return {
        "best_n_components": best_n,
        "MAE":               mae,
        "R2":                r2,
        "pearson_r":         corr,
        "pearson_p":         pval
    }

# --------------------------------------------------
# Load and merge data
# --------------------------------------------------
df_t1 = pd.read_csv("/blue/neurology-dept/JOSH/_UF_DBS/T1_synthseg_vols_robust_no_parc.csv")
crs   = pd.read_csv("CRS_labels.csv").dropna()

df_t1["subject_id"] = df_t1["subject"].str.extract(r"(\d+)").astype(int)
df = df_t1.merge(crs, left_on="subject_id", right_on="ID")
print(f"Subjects after merge: {len(df)}")

y           = df["neuropsych_score"].values
df_features = df[["total intracranial"] + brain_cols].copy()

# Feature name lists for coefficient inspection
resid_feature_names    = [f"resid_{c}" for c in brain_cols]
combined_feature_names = brain_cols + resid_feature_names

# --------------------------------------------------
# Nested CV setup
# --------------------------------------------------
outer_cv = KFold(n_splits=2, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=2, shuffle=True, random_state=42)

elasticnet_param_grid = {
    "alpha":    [0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.1, 0.5, 0.9]
}

conditions = [
    "original_only",
    "single_residual_only",
    "single_original_residual",
    "bootstrap_residual_only",
    "bootstrap_original_residual"
]

all_results_en  = {c: [] for c in conditions}  # ElasticNet results
all_results_pls = {c: [] for c in conditions}  # PLS results

# Collect ElasticNet coefficients for all conditions
# After results are in, inspection runs on whichever condition performed best
coef_inspection = {c: [] for c in conditions}

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
    res_en  = evaluate_elasticnet(X_train, X_test, y_train, y_test,
                                  elasticnet_param_grid, inner_cv, brain_cols)
    res_pls = evaluate_pls(X_train, X_test, y_train, y_test, inner_cv)
    res_en.update({"fold": fold_idx+1, "n_train": len(y_train), "n_test": len(y_test)})
    res_pls.update({"fold": fold_idx+1})
    all_results_en["original_only"].append(res_en)
    all_results_pls["original_only"].append(res_pls)
    coef_inspection["original_only"].append({"fold": fold_idx+1, "coef": res_en["coef"], "feature_names": brain_cols})
    print(f"  EN:  MAE={res_en['MAE']:.4f}  R²={res_en['R2']:.4f}  r={res_en['pearson_r']:.4f}")
    print(f"  PLS: MAE={res_pls['MAE']:.4f}  R²={res_pls['R2']:.4f}  r={res_pls['pearson_r']:.4f}  n_comp={res_pls['best_n_components']}")

    # --------------------------------------------------
    # Fit single-run DAG + normalized residuals
    # --------------------------------------------------
    print("\n  Fitting single-run DirectLiNGAM...")
    adj_single = fit_dag_single(X_train)
    R_train_single = compute_causal_residuals(X_train, adj_single)
    R_test_single  = compute_causal_residuals(X_test,  adj_single)
    Z_train_single, Z_test_single = normalize_residuals(R_train_single, R_test_single)

    # --------------------------------------------------
    # Condition 2: Single DAG — residuals only
    # --------------------------------------------------
    print("\n[2/5] Single DAG — residuals only...")
    res_en  = evaluate_elasticnet(Z_train_single, Z_test_single, y_train, y_test,
                                  elasticnet_param_grid, inner_cv, resid_feature_names)
    res_pls = evaluate_pls(Z_train_single, Z_test_single, y_train, y_test, inner_cv)
    res_en.update({"fold": fold_idx+1, "n_train": len(y_train), "n_test": len(y_test)})
    res_pls.update({"fold": fold_idx+1})
    all_results_en["single_residual_only"].append(res_en)
    all_results_pls["single_residual_only"].append(res_pls)
    coef_inspection["single_residual_only"].append({"fold": fold_idx+1, "coef": res_en["coef"], "feature_names": resid_feature_names})
    print(f"  EN:  MAE={res_en['MAE']:.4f}  R²={res_en['R2']:.4f}  r={res_en['pearson_r']:.4f}")
    print(f"  PLS: MAE={res_pls['MAE']:.4f}  R²={res_pls['R2']:.4f}  r={res_pls['pearson_r']:.4f}  n_comp={res_pls['best_n_components']}")

    # --------------------------------------------------
    # Condition 3: Single DAG — original + residuals
    # --------------------------------------------------
    print("\n[3/5] Single DAG — original + residuals...")
    X_tr_c = np.hstack([X_train, Z_train_single])
    X_te_c = np.hstack([X_test,  Z_test_single])
    res_en  = evaluate_elasticnet(X_tr_c, X_te_c, y_train, y_test,
                                  elasticnet_param_grid, inner_cv, combined_feature_names)
    res_pls = evaluate_pls(X_tr_c, X_te_c, y_train, y_test, inner_cv)
    res_en.update({"fold": fold_idx+1, "n_train": len(y_train), "n_test": len(y_test)})
    res_pls.update({"fold": fold_idx+1})
    all_results_en["single_original_residual"].append(res_en)
    all_results_pls["single_original_residual"].append(res_pls)
    coef_inspection["single_original_residual"].append({"fold": fold_idx+1, "coef": res_en["coef"], "feature_names": combined_feature_names})
    print(f"  EN:  MAE={res_en['MAE']:.4f}  R²={res_en['R2']:.4f}  r={res_en['pearson_r']:.4f}")
    print(f"  PLS: MAE={res_pls['MAE']:.4f}  R²={res_pls['R2']:.4f}  r={res_pls['pearson_r']:.4f}  n_comp={res_pls['best_n_components']}")

    # --------------------------------------------------
    # Fit bootstrapped DAG + normalized residuals
    # --------------------------------------------------
    print("\n  Fitting bootstrapped DirectLiNGAM (100 runs)...")
    adj_boot = fit_dag_bootstrap(X_train, n_bootstrap=100, threshold=0.5)
    R_train_boot = compute_causal_residuals(X_train, adj_boot)
    R_test_boot  = compute_causal_residuals(X_test,  adj_boot)
    Z_train_boot, Z_test_boot = normalize_residuals(R_train_boot, R_test_boot)

    # --------------------------------------------------
    # Condition 4: Bootstrap DAG — residuals only
    # --------------------------------------------------
    print("\n[4/5] Bootstrap DAG — residuals only...")
    res_en  = evaluate_elasticnet(Z_train_boot, Z_test_boot, y_train, y_test,
                                  elasticnet_param_grid, inner_cv, resid_feature_names)
    res_pls = evaluate_pls(Z_train_boot, Z_test_boot, y_train, y_test, inner_cv)
    res_en.update({"fold": fold_idx+1, "n_train": len(y_train), "n_test": len(y_test)})
    res_pls.update({"fold": fold_idx+1})
    all_results_en["bootstrap_residual_only"].append(res_en)
    all_results_pls["bootstrap_residual_only"].append(res_pls)
    coef_inspection["bootstrap_residual_only"].append({"fold": fold_idx+1, "coef": res_en["coef"], "feature_names": resid_feature_names})
    print(f"  EN:  MAE={res_en['MAE']:.4f}  R²={res_en['R2']:.4f}  r={res_en['pearson_r']:.4f}")
    print(f"  PLS: MAE={res_pls['MAE']:.4f}  R²={res_pls['R2']:.4f}  r={res_pls['pearson_r']:.4f}  n_comp={res_pls['best_n_components']}")

    # --------------------------------------------------
    # Condition 5: Bootstrap DAG — original + residuals (BEST CONDITION)
    # Also collect ElasticNet coefficients for inspection
    # --------------------------------------------------
    print("\n[5/5] Bootstrap DAG — original + residuals...")
    X_tr_cb = np.hstack([X_train, Z_train_boot])
    X_te_cb = np.hstack([X_test,  Z_test_boot])
    res_en  = evaluate_elasticnet(X_tr_cb, X_te_cb, y_train, y_test,
                                  elasticnet_param_grid, inner_cv, combined_feature_names)
    res_pls = evaluate_pls(X_tr_cb, X_te_cb, y_train, y_test, inner_cv)
    res_en.update({"fold": fold_idx+1, "n_train": len(y_train), "n_test": len(y_test)})
    res_pls.update({"fold": fold_idx+1})
    all_results_en["bootstrap_original_residual"].append(res_en)
    all_results_pls["bootstrap_original_residual"].append(res_pls)
    coef_inspection["bootstrap_original_residual"].append({"fold": fold_idx+1, "coef": res_en["coef"], "feature_names": combined_feature_names})
    print(f"  EN:  MAE={res_en['MAE']:.4f}  R²={res_en['R2']:.4f}  r={res_en['pearson_r']:.4f}")
    print(f"  PLS: MAE={res_pls['MAE']:.4f}  R²={res_pls['R2']:.4f}  r={res_pls['pearson_r']:.4f}  n_comp={res_pls['best_n_components']}")

# --------------------------------------------------
# Summary: ElasticNet
# --------------------------------------------------
print(f"\n{'='*60}")
print("FINAL SUMMARY — ElasticNet")
print(f"{'='*60}")

en_rows = []
for condition, fold_results in all_results_en.items():
    maes = [r["MAE"] for r in fold_results]
    r2s  = [r["R2"]  for r in fold_results]
    rs   = [r["pearson_r"] for r in fold_results]
    en_rows.append({
        "condition":      condition,
        "mean_MAE":       np.mean(maes),
        "std_MAE":        np.std(maes),
        "mean_R2":        np.mean(r2s),
        "std_R2":         np.std(r2s),
        "mean_pearson_r": np.nanmean(rs),
        "std_pearson_r":  np.nanstd(rs),
    })

en_df = pd.DataFrame(en_rows)
print(en_df.to_string(index=False))
en_df.to_csv("nested_cv_elasticnet_results.csv", index=False)
print("\nSaved: nested_cv_elasticnet_results.csv")

# --------------------------------------------------
# Summary: PLS
# --------------------------------------------------
print(f"\n{'='*60}")
print("FINAL SUMMARY — PLSRegression")
print(f"{'='*60}")

pls_rows = []
for condition, fold_results in all_results_pls.items():
    maes   = [r["MAE"] for r in fold_results]
    r2s    = [r["R2"]  for r in fold_results]
    rs     = [r["pearson_r"] for r in fold_results]
    ncomps = [r["best_n_components"] for r in fold_results]
    pls_rows.append({
        "condition":      condition,
        "mean_MAE":       np.mean(maes),
        "std_MAE":        np.std(maes),
        "mean_R2":        np.mean(r2s),
        "std_R2":         np.std(r2s),
        "mean_pearson_r": np.nanmean(rs),
        "std_pearson_r":  np.nanstd(rs),
        "mean_n_components": np.mean(ncomps)
    })

pls_df = pd.DataFrame(pls_rows)
print(pls_df.to_string(index=False))
pls_df.to_csv("nested_cv_pls_results.csv", index=False)
print("\nSaved: nested_cv_pls_results.csv")

# --------------------------------------------------
# ElasticNet coefficient inspection — all conditions
# After reviewing results, focus on whichever condition performed best
# --------------------------------------------------
print(f"\n{'='*60}")
print("COEFFICIENT INSPECTION — All Conditions (ElasticNet)")
print(f"{'='*60}")

for condition, entries in coef_inspection.items():
    if not entries:
        continue

    feature_names = entries[0]["feature_names"]
    n_features    = len(feature_names)
    coef_matrix   = np.zeros((len(entries), n_features))
    for i, entry in enumerate(entries):
        coef_matrix[i, :] = entry["coef"]

    coef_df = pd.DataFrame(
        coef_matrix,
        columns=feature_names,
        index=[f"fold_{e['fold']}" for e in entries]
    )
    coef_df.loc["mean"] = coef_df.mean()
    coef_df.loc["std"]  = coef_df.std()

    # Nonzero in at least one fold
    nonzero_mask    = (coef_matrix != 0).any(axis=0)
    active_features = np.array(feature_names)[nonzero_mask]

    # Stable: nonzero in ALL folds
    stable_mask    = (coef_matrix != 0).all(axis=0)
    stable_features = np.array(feature_names)[stable_mask]

    # Residual nodes specifically (only relevant for conditions with residuals)
    resid_mask    = np.array(["resid_" in f for f in feature_names])
    resid_nonzero = nonzero_mask & resid_mask

    print(f"\n--- {condition} ---")
    print(f"  Active features (nonzero in >=1 fold): {nonzero_mask.sum()}/{n_features}")
    print(f"  Stable features (nonzero in all folds): {stable_mask.sum()}/{n_features}")
    if resid_mask.any():
        print(f"  Residual nodes surviving: {resid_nonzero.sum()}/{resid_mask.sum()}")

    if nonzero_mask.sum() > 0:
        active_df = coef_df[active_features].T.copy()
        active_df["is_residual"] = ["resid_" in f for f in active_df.index]
        print(active_df.to_string())

    # Save per-condition CSV
    coef_df.to_csv(f"elasticnet_coef_{condition}.csv")
    print(f"  Saved: elasticnet_coef_{condition}.csv")
