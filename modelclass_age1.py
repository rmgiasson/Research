import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             confusion_matrix)
import lingam
import warnings
import sys
import os
warnings.filterwarnings("ignore")

repeat_idx  = int(sys.argv[1])
random_seed = 42 + repeat_idx

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

# Age is index 0 in the DAG feature matrix [age | brain_cols]
# Education goes only into the classifier
AGE_IDX  = 0
dag_cols = ["age"] + brain_cols

# --------------------------------------------------
# Build prior knowledge matrix for DirectLiNGAM
# prior[i,j]=1  means x_i -> x_j is allowed
# prior[i,j]=0  means x_i -> x_j is forbidden
# prior[i,j]=-1 means no prior knowledge
# Age (index 0) can cause brain regions but not vice versa
# --------------------------------------------------
def build_prior(n_features):
    prior = np.full((n_features, n_features), -1)
    for j in range(1, n_features):
        prior[AGE_IDX, j] = 0   # age -> brain_region: allowed
        prior[j, AGE_IDX] = 1   # brain_region -> age: forbidden
    return prior

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
    icv_coefs    = fit_icv_residualizer(df_train)
    resid_train  = apply_icv_residualize(df_train, icv_coefs)
    resid_test   = apply_icv_residualize(df_test,  icv_coefs)
    brain_scaler = StandardScaler()
    X_brain_train = brain_scaler.fit_transform(resid_train)
    X_brain_test  = brain_scaler.transform(resid_test)
    age_scaler   = StandardScaler()
    X_age_train  = age_scaler.fit_transform(
        df_train[["age"]].values.astype(float))
    X_age_test   = age_scaler.transform(
        df_test[["age"]].values.astype(float))
    edu_scaler   = StandardScaler()
    X_edu_train  = edu_scaler.fit_transform(
        df_train[["education"]].values.astype(float))
    X_edu_test   = edu_scaler.transform(
        df_test[["education"]].values.astype(float))
    X_dag_train  = np.hstack([X_age_train, X_brain_train])
    X_dag_test   = np.hstack([X_age_test,  X_brain_test])
    return (X_brain_train, X_brain_test,
            X_edu_train,   X_edu_test,
            X_dag_train,   X_dag_test)

# --------------------------------------------------
# Bootstrap DAG with age constrained as upstream
# via prior_knowledge during estimation — not post-hoc
# --------------------------------------------------
def fit_dag_bootstrap_age_upstream(X_dag_train, n_bootstrap=100,
                                   threshold=0.5, random_state=42):
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X_dag_train.shape
    prior = build_prior(n_features)
    weight_sums = np.zeros((n_features, n_features))
    edge_counts = np.zeros((n_features, n_features))
    successful_runs = 0

    for b in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        try:
            model = lingam.DirectLiNGAM(
                random_state=b,
                prior_knowledge=prior,
                apply_prior_knowledge_softly=False  # hard constraint
            )
            model.fit(X_dag_train[idx])
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
    edge_probs   = edge_counts / max(successful_runs, 1)
    consensus_adj = np.where(edge_probs >= threshold, mean_weights, 0.0)

    n_age_children = int((consensus_adj[1:, AGE_IDX] != 0).sum())
    print(f"    Bootstrap done: {successful_runs}/{n_bootstrap} runs, "
          f"{n_age_children}/{len(brain_cols)} brain regions "
          f"have age as parent")
    return consensus_adj

def compute_causal_residuals_brain_only(X_dag, adj):
    """Residuals for brain regions only — age (index 0) can be a parent."""
    n_samples = X_dag.shape[0]
    n_brain   = len(brain_cols)
    residuals = np.zeros((n_samples, n_brain))
    for i_brain in range(n_brain):
        i_dag   = i_brain + 1
        parents = np.where(adj[i_dag, :] != 0)[0]
        if len(parents) == 0:
            residuals[:, i_brain] = X_dag[:, i_dag]
        else:
            residuals[:, i_brain] = (X_dag[:, i_dag]
                                     - X_dag[:, parents] @ adj[i_dag, parents])
    return residuals

def normalize_residuals(R_train, R_test):
    sigma = R_train.std(axis=0, ddof=1)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return R_train / sigma, R_test / sigma

# --------------------------------------------------
# Classifier — StratifiedKFold for balanced class splits
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
        param_grid, cv=inner_cv, scoring="roc_auc",
        refit=True, n_jobs=1
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
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {"balanced_acc": bal_acc, "auc": auc,
            "sensitivity": sensitivity, "specificity": specificity}

# --------------------------------------------------
# Load data
# --------------------------------------------------
df_t1 = pd.read_csv("T1_synthseg_vols_robust_no_parc.csv")
crs   = pd.read_csv("CRS_labels_new.csv")
crs   = crs.dropna(subset=["neuropsych_score", "age", "education"])
df_t1["subject_id"] = df_t1["subject"].str.extract(r"(\d+)").astype(int)
df    = df_t1.merge(crs, left_on="subject_id", right_on="ID")
df["label"] = (df["neuropsych_score"] >= 3).astype(int)

if repeat_idx == 0:
    print(f"Subjects after merge: {len(df)}")
    print(f"GO (score<3):     {(df['label']==0).sum()}")
    print(f"NO-GO (score>=3): {(df['label']==1).sum()}")

y        = df["label"].values
df_brain = df[["total intracranial"] + brain_cols].copy()
df_meta  = df[["age", "education"]].copy()

# StratifiedKFold preserves class proportions across folds
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

fold_rows = []

for fold_idx, (train_idx, test_idx) in enumerate(
        outer_cv.split(df_brain, y)):

    df_brain_train = df_brain.iloc[train_idx].reset_index(drop=True)
    df_brain_test  = df_brain.iloc[test_idx].reset_index(drop=True)
    df_meta_train  = df_meta.iloc[train_idx].reset_index(drop=True)
    df_meta_test   = df_meta.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test  = y[test_idx]

    (X_brain_train, X_brain_test,
     X_edu_train,   X_edu_test,
     X_dag_train,   X_dag_test) = preprocess_train_test(
        pd.concat([df_brain_train, df_meta_train], axis=1),
        pd.concat([df_brain_test,  df_meta_test],  axis=1)
    )

    print(f"  Fold {fold_idx+1} — fitting bootstrap DAG with age upstream...")
    adj = fit_dag_bootstrap_age_upstream(
        X_dag_train, n_bootstrap=100,
        threshold=0.5, random_state=random_seed
    )
    brain_to_age = np.sum(adj[AGE_IDX, 1:] != 0)
    print(f"Brain -> Age edges: {brain_to_age}")
    R_tr = compute_causal_residuals_brain_only(X_dag_train, adj)
    R_te = compute_causal_residuals_brain_only(X_dag_test,  adj)
    Z_tr, Z_te = normalize_residuals(R_tr, R_te)

    # Conditions: education in classifier, age in DAG only
    for cond_name, X_tr, X_te in [
        ("original_edu",           # brain volumes + education
         np.hstack([X_brain_train, X_edu_train]),
         np.hstack([X_brain_test,  X_edu_test])),
        ("residual_age_dag_edu",   # age-informed residuals + education
         np.hstack([Z_tr, X_edu_train]),
         np.hstack([Z_te, X_edu_test])),
        ("original_residual_age_dag_edu",  # all three
         np.hstack([X_brain_train, Z_tr, X_edu_train]),
         np.hstack([X_brain_test,  Z_te, X_edu_test])),
    ]:
        res = evaluate_classifier(X_tr, X_te, y_train, y_test, inner_cv)
        res.update({"repeat": repeat_idx, "fold": fold_idx,
                    "condition": cond_name})
        fold_rows.append(res)

out_df = pd.DataFrame(fold_rows)
os.makedirs("classification_age_dag_31_results", exist_ok=True)
out_df.to_csv(
    f"classification_age_dag_31_results/repeat_{repeat_idx:02d}.csv",
    index=False)
print(f"Repeat {repeat_idx} done.")
for cond in ["original_edu", "residual_age_dag_edu",
             "original_residual_age_dag_edu"]:
    sub = out_df[out_df["condition"] == cond]
    print(f"  {cond}: "
          f"bal_acc={sub['balanced_acc'].mean():.4f}  "
          f"auc={sub['auc'].mean():.4f}  "
          f"sens={sub['sensitivity'].mean():.4f}  "
          f"spec={sub['specificity'].mean():.4f}")
