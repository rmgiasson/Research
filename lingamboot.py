import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr
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
# Preprocessing: ICV residualization + standardization
# --------------------------------------------------
def preprocess(df):
    """ICV residualize then standardize. Returns numpy array."""
    df = df.drop(columns=["subject", "patno", "subject_id"], errors="ignore")
    df = df[["total intracranial"] + brain_cols].copy()
    icv = df["total intracranial"].values.reshape(-1, 1)
    residuals = np.zeros((len(df), len(brain_cols)))
    for i, col in enumerate(brain_cols):
        y = df[col].values
        reg = LinearRegression()
        reg.fit(icv, y)
        residuals[:, i] = y - reg.predict(icv)
    scaler = StandardScaler()
    return scaler.fit_transform(residuals)

# --------------------------------------------------
# Bootstrap DirectLiNGAM
# Returns full edge inclusion probability matrix (no thresholding)
# adj[i,j] = probability that edge j->i exists across bootstrap runs
# --------------------------------------------------
def fit_dag_bootstrap(X, n_bootstrap=100, random_state=42):
    """
    Fit DirectLiNGAM on n_bootstrap subsamples.
    Returns full edge inclusion probability matrix (values 0-1).
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape
    edge_counts = np.zeros((n_features, n_features))
    successful_runs = 0

    for b in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[idx]
        try:
            model = lingam.DirectLiNGAM(random_state=b)
            model.fit(X_boot)
            adj = model.adjacency_matrix_
            edge_counts += (np.abs(adj) > 0).astype(float)
            successful_runs += 1
        except Exception:
            continue

    print(f"  Successful bootstrap runs: {successful_runs}/{n_bootstrap}")
    edge_probs = edge_counts / successful_runs
    return edge_probs

# --------------------------------------------------
# Matrix similarity metrics
# Comparing full probability matrices without thresholding
# --------------------------------------------------
def compare_matrices(A, B, name_a, name_b):
    """
    Compare two DAG probability matrices on structure and weight similarity.
    A, B: (n_features x n_features) edge inclusion probability matrices
    """
    print(f"\n  --- Similarity: {name_a} vs {name_b} ---")

    # Flatten to vectors, exclude diagonal
    mask = ~np.eye(A.shape[0], dtype=bool)
    a_flat = A[mask]
    b_flat = B[mask]

    # 1. Frobenius norm of difference (lower = more similar)
    frob = np.linalg.norm(A - B, 'fro')
    print(f"  Frobenius norm of difference:     {frob:.4f}  (lower = more similar)")

    # 2. Pearson correlation of edge weights
    pearson_r, pearson_p = pearsonr(a_flat, b_flat)
    print(f"  Pearson correlation of weights:   r={pearson_r:.4f}  (p={pearson_p:.4f})")

    # 3. Spearman correlation (rank-based, robust to outlier edges)
    spearman_r, spearman_p = spearmanr(a_flat, b_flat)
    print(f"  Spearman correlation of weights:  r={spearman_r:.4f}  (p={spearman_p:.4f})")

    # 4. Weighted Jaccard similarity
    # For continuous values: sum(min(a,b)) / sum(max(a,b))
    numerator   = np.sum(np.minimum(a_flat, b_flat))
    denominator = np.sum(np.maximum(a_flat, b_flat))
    weighted_jaccard = numerator / denominator if denominator > 0 else 0.0
    print(f"  Weighted Jaccard similarity:      {weighted_jaccard:.4f}  (higher = more similar, max=1)")

    # 5. Mean absolute difference of edge probabilities
    mad = np.mean(np.abs(a_flat - b_flat))
    print(f"  Mean absolute difference:         {mad:.4f}  (lower = more similar)")

    return {
        "comparison":       f"{name_a}_vs_{name_b}",
        "frobenius_norm":   frob,
        "pearson_r":        pearson_r,
        "pearson_p":        pearson_p,
        "spearman_r":       spearman_r,
        "spearman_p":       spearman_p,
        "weighted_jaccard": weighted_jaccard,
        "mean_abs_diff":    mad
    }

# --------------------------------------------------
# Run bootstrap LiNGAM and save results
# --------------------------------------------------
def run_bootstrap_lingam(name, df, n_bootstrap=100):
    print(f"\n{'='*50}")
    print(f"Running Bootstrap DirectLiNGAM on: {name} ({len(df)} subjects)")
    print(f"{'='*50}")

    X = preprocess(df.copy())
    adj_prob = fit_dag_bootstrap(X, n_bootstrap=n_bootstrap)

    # Save full probability matrix as labeled CSV
    W_df = pd.DataFrame(adj_prob, index=brain_cols, columns=brain_cols)
    W_df.to_csv(f"bootstrap_lingam_{name}.csv")
    print(f"  Saved: bootstrap_lingam_{name}.csv")

    # Print edges above 0.5 inclusion probability
    threshold = 0.5
    print(f"\n  Edges with inclusion probability > {threshold}:")
    found = False
    for i in range(len(brain_cols)):
        for j in range(len(brain_cols)):
            if adj_prob[i, j] > threshold:
                print(f"    {brain_cols[j]} -> {brain_cols[i]}: {adj_prob[i, j]:.3f}")
                found = True
    if not found:
        print("    None found above threshold")

    return adj_prob

# --------------------------------------------------
# Load datasets
# --------------------------------------------------
df_t1   = pd.read_csv("/blue/neurology-dept/JOSH/_UF_DBS/T1_synthseg_vols_robust_no_parc.csv")
df_ppmi = pd.read_csv("PPMI_all_patients_brain_imaging.csv")

# --------------------------------------------------
# Load CRS labels and split T1 by diagnosis
# --------------------------------------------------
crs = pd.read_csv("CRS_labels_new.csv").dropna()
df_t1["subject_id"] = df_t1["subject"].str.extract(r"(\d+)").astype(int)
df_t1_crs = df_t1.merge(crs, left_on="subject_id", right_on="ID")
print(f"T1 matched with CRS: {len(df_t1_crs)} subjects")

df_t1_pd    = df_t1_crs[df_t1_crs["diagnosis"] == 1].copy()
df_t1_nonpd = df_t1_crs[df_t1_crs["diagnosis"] != 1].copy()
print(f"T1 PD only:  {len(df_t1_pd)} subjects")
print(f"T1 non-PD:   {len(df_t1_nonpd)} subjects")
print(f"PPMI:        {len(df_ppmi)} subjects")

# --------------------------------------------------
# Run bootstrap LiNGAM on all groups
# --------------------------------------------------
adj_ppmi     = run_bootstrap_lingam("PPMI",     df_ppmi)
adj_t1_pd    = run_bootstrap_lingam("T1_PD",    df_t1_pd)
adj_t1_nonpd = run_bootstrap_lingam("T1_nonPD", df_t1_nonpd)

# --------------------------------------------------
# Compare matrices
# Key comparison: does T1_PD look more like PPMI than T1_nonPD does?
# --------------------------------------------------
print(f"\n{'='*50}")
print("MATRIX SIMILARITY COMPARISONS")
print(f"{'='*50}")

results = []
results.append(compare_matrices(adj_ppmi, adj_t1_pd,    "PPMI", "T1_PD"))
results.append(compare_matrices(adj_ppmi, adj_t1_nonpd, "PPMI", "T1_nonPD"))
results.append(compare_matrices(adj_t1_pd, adj_t1_nonpd, "T1_PD", "T1_nonPD"))

# --------------------------------------------------
# Summary: is T1_PD more similar to PPMI than T1_nonPD?
# --------------------------------------------------
print(f"\n{'='*50}")
print("SUMMARY: Is T1_PD more similar to PPMI than T1_nonPD?")
print(f"{'='*50}")

ppmi_vs_pd    = results[0]
ppmi_vs_nonpd = results[1]

metrics = [
    ("Frobenius norm",    "frobenius_norm",   "lower"),
    ("Pearson r",         "pearson_r",        "higher"),
    ("Spearman r",        "spearman_r",       "higher"),
    ("Weighted Jaccard",  "weighted_jaccard", "higher"),
    ("Mean abs diff",     "mean_abs_diff",    "lower"),
]

pd_more_similar = 0
for label, key, direction in metrics:
    pd_val    = ppmi_vs_pd[key]
    nonpd_val = ppmi_vs_nonpd[key]
    if direction == "lower":
        pd_wins = pd_val < nonpd_val
    else:
        pd_wins = pd_val > nonpd_val
    winner = "T1_PD more similar" if pd_wins else "T1_nonPD more similar"
    pd_more_similar += int(pd_wins)
    print(f"  {label:20s}  PPMI vs PD={pd_val:.4f}  PPMI vs nonPD={nonpd_val:.4f}  -> {winner}")

print(f"\n  T1_PD was more similar to PPMI on {pd_more_similar}/5 metrics")

# Save similarity results
results_df = pd.DataFrame(results)
results_df.to_csv("dag_similarity_results.csv", index=False)
print("\nSaved: dag_similarity_results.csv")
