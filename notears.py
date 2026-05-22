import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

# --------------------------------------------------
# NOTEARS functions
# --------------------------------------------------
def acyclicity_constraint(W, d):
    expm = torch.matrix_exp(W * W)
    return torch.trace(expm) - d

def run_notears(X, d, lambda1=0.01, max_outer=20, inner_epochs=500, lr=0.01,
                rho_init=1.0, rho_max=1e16, h_tol=1e-8):
    W = torch.zeros((d, d), requires_grad=True)
    optimizer = torch.optim.Adam([W], lr=lr)
    rho = rho_init
    alpha = 0.0
    h_prev = np.inf

    for outer in range(max_outer):
        for epoch in range(inner_epochs):
            optimizer.zero_grad()
            X_hat = X @ W
            loss = ((X - X_hat) ** 2).mean()
            h = acyclicity_constraint(W, d)
            obj = (loss
                   + lambda1 * torch.sum(torch.abs(W))
                   + 0.5 * rho * h * h
                   + alpha * h)
            obj.backward()
            optimizer.step()
            with torch.no_grad():
                W.fill_diagonal_(0)

        h_val = acyclicity_constraint(W, d).item()
        loss_val = ((X - X @ W) ** 2).mean().item()
        print(f"  Outer {outer+1:02d} | Loss: {loss_val:.4f} | h(W): {h_val:.2e} | rho: {rho:.1e} | alpha: {alpha:.4f}")

        if h_val <= h_tol:
            print("  DAG constraint satisfied!")
            break

        if h_val > 0.25 * h_prev:
            rho = min(rho * 10, rho_max)
        alpha += rho * h_val
        h_prev = h_val

    return W

# --------------------------------------------------
# Shared brain volume columns
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
    "csf",
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

cols_to_keep = ["total intracranial"] + brain_cols

# --------------------------------------------------
# Helper: normalize by ICV
# --------------------------------------------------
def normalize_icv(df):
    df = df.drop(columns=["subject", "pat", "patno", "subject_id", "ID",
                           "diagnosis", "neuropsych_score", "record_id",
                           "sex", "education", "neuropsych_drs", "age"],
                 errors="ignore")
    df = df[cols_to_keep]
    icv = df["total intracranial"].values
    df_norm = df[brain_cols].div(icv, axis=0)
    return df_norm

# --------------------------------------------------
# Helper: preprocess for NOTEARS
# --------------------------------------------------
def preprocess(df_norm):
    scaler = StandardScaler()
    X = scaler.fit_transform(df_norm.values)
    X = torch.tensor(X, dtype=torch.float32)
    return X, df_norm.columns.tolist()

# --------------------------------------------------
# Helper: run and save
# --------------------------------------------------
def run_and_save(name, df):
    print(f"\n{'='*50}")
    print(f"Running NOTEARS on: {name}")
    print(f"{'='*50}")
    df_norm = normalize_icv(df.copy())
    X, col_names = preprocess(df_norm)
    d = X.shape[1]
    W = run_notears(X, d)

    W_est = W.detach().numpy()
    np.fill_diagonal(W_est, 0)

    W_df = pd.DataFrame(W_est, index=col_names, columns=col_names)
    W_df.to_csv(f"adjacency_matrix_{name}.csv")
    print(f"\nSaved: adjacency_matrix_{name}.csv")

    threshold = 0.2
    print(f"\nEdges with |weight| > {threshold}:")
    rows, cols = np.where(np.abs(W_est) > threshold)
    if len(rows) == 0:
        print("  None found above threshold")
    for r, c in zip(rows, cols):
        print(f"  {col_names[r]} → {col_names[c]}: {W_est[r, c]:.4f}")

# --------------------------------------------------
# Load datasets
# --------------------------------------------------
df_t1   = pd.read_csv("/blue/neurology-dept/JOSH/_UF_DBS/T1_synthseg_vols_robust_no_parc.csv")
df_ppmi = pd.read_csv("PPMI_all_patients_brain_imaging.csv")  # update path
crs     = pd.read_csv("CRS_labels.csv")

# --------------------------------------------------
# Extract T1 PD-only subjects (diagnosis == 1)
# --------------------------------------------------
crs_clean = crs.dropna()
df_t1["subject_id"] = df_t1["subject"].str.extract(r"(\d+)").astype(int)
df_t1_merged = df_t1.merge(crs_clean, left_on="subject_id", right_on="ID")
df_t1_pd = df_t1_merged[df_t1_merged["diagnosis"] == 1].copy()
print(f"T1 PD-only subjects: {len(df_t1_pd)}")

# --------------------------------------------------
# Combine T1 PD-only with PPMI
# --------------------------------------------------
df_combined_pd = pd.concat([df_ppmi, df_t1_pd], axis=0, ignore_index=True)
print(f"PPMI + T1 PD-only subjects: {len(df_combined_pd)}")

# --------------------------------------------------
# Run NOTEARS
# --------------------------------------------------
run_and_save("PPMI", df_ppmi)
run_and_save("PPMI_plus_T1_PD", df_combined_pd)
