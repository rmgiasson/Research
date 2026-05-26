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
resid_names = [f"resid_{c}" for c in brain_cols]

# --------------------------------------------------
# Preprocessing (ICV + scaling)
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
# ORIGINAL feature preprocessing (NEW)
# --------------------------------------------------
def preprocess_original(df_train, df_test):
    X_train = df_train[brain_cols].values
    X_test  = df_test[brain_cols].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
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

    return mae, r2, corr, model.coef_

# --------------------------------------------------
# Load data
# --------------------------------------------------
df_t1 = pd.read_csv("/blue/neurology-dept/JOSH/_UF_DBS/T1_synthseg_vols_robust_no_parc.csv")
crs   = pd.read_csv("CRS_labels.csv").dropna()

df_t1["subject_id"] = df_t1["subject"].str.extract(r"(\d+)").astype(int)
df    = df_t1.merge(crs, left_on="subject_id", right_on="ID")

y = df["neuropsych_score"].values

df_features = df[brain_cols].copy()

# --------------------------------------------------
# CV setup
# --------------------------------------------------
outer_cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)

perf_rows = []
coef_rows = []

# --------------------------------------------------
# CV LOOP
# --------------------------------------------------
for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(df_features)):

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    y_train = y[train_idx]
    y_test  = y[test_idx]

    # ==================================================
    # ORIGINAL MODEL
    # ==================================================
    X_train_o, X_test_o = preprocess_original(df_train, df_test)

    mae_o, r2_o, r_o, coef_o = evaluate_elasticnet(
        X_train_o, X_test_o, y_train, y_test, inner_cv
    )

    perf_rows.append({
        "repeat": repeat_idx,
        "fold": fold_idx,
        "model": "original",
        "MAE": mae_o,
        "R2": r2_o,
        "pearson_r": r_o
    })

    for feat, c in zip(brain_cols, coef_o):
        coef_rows.append({
            "repeat": repeat_idx,
            "fold": fold_idx,
            "model": "original",
            "feature": feat,
            "coef": c,
            "nonzero": int(c != 0),
            "sign": int(np.sign(c))
        })

    # ==================================================
    # RESIDUAL MODEL (DAG)
    # ==================================================
    X_train_r, X_test_r = preprocess_train_test(df_train, df_test)

    adj  = fit_dag_bootstrap(X_train_r, n_bootstrap=100,
                             threshold=0.5,
                             random_state=random_seed)

    R_tr = compute_causal_residuals(X_train_r, adj)
    R_te = compute_causal_residuals(X_test_r, adj)

    Z_tr, Z_te = normalize_residuals(R_tr, R_te)

    mae_r, r2_r, r_r, coef_r = evaluate_elasticnet(
        Z_tr, Z_te, y_train, y_test, inner_cv
    )

    perf_rows.append({
        "repeat": repeat_idx,
        "fold": fold_idx,
        "model": "residual",
        "MAE": mae_r,
        "R2": r2_r,
        "pearson_r": r_r
    })

    for feat, c in zip(resid_names, coef_r):
        coef_rows.append({
            "repeat": repeat_idx,
            "fold": fold_idx,
            "model": "residual",
            "feature": feat,
            "coef": c,
            "nonzero": int(c != 0),
            "sign": int(np.sign(c))
        })

# --------------------------------------------------
# SAVE
# --------------------------------------------------
os.makedirs("coef_results", exist_ok=True)

pd.DataFrame(perf_rows).to_csv(
    f"coef_results/perf_{repeat_idx:02d}.csv",
    index=False
)

pd.DataFrame(coef_rows).to_csv(
    f"coef_results/coef_{repeat_idx:02d}.csv",
    index=False
)

print(f"Repeat {repeat_idx} done")
