import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lingam
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# Brain volume columns
# ============================================================

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

cols_to_keep = ["total intracranial"] + brain_cols


# ============================================================
# ICV normalization
# ============================================================

def normalize_icv(df, name="dataset"):

    # Make sure all required columns exist
    missing_cols = [
        col for col in cols_to_keep
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"{name} is missing required columns:\n"
            + "\n".join(missing_cols)
        )

    # Keep only ICV + brain regions
    df = df[cols_to_keep].copy()

    before = len(df)

    # Remove rows with missing values
    df = df.dropna()

    after_nan = len(df)

    # Remove invalid ICV values
    df = df[df["total intracranial"] > 0].copy()

    after_icv = len(df)

    print(f"\n{name} preprocessing:")
    print(f"  Original subjects:         {before}")
    print(f"  After NaN removal:         {after_nan}")
    print(f"  After invalid ICV removal: {after_icv}")

    if after_icv == 0:
        raise ValueError(
            f"No valid subjects remain in {name}."
        )

    # Divide each regional volume by ICV
    icv = df["total intracranial"]

    df_norm = df[brain_cols].div(icv, axis=0)

    if not np.isfinite(df_norm.values).all():
        raise ValueError(
            f"{name} contains non-finite values "
            "after ICV normalization."
        )

    return df_norm


# ============================================================
# Standardization
# ============================================================

def preprocess(df_norm):

    scaler = StandardScaler()

    X = scaler.fit_transform(df_norm.values)

    X = np.asarray(X, dtype=np.float64)

    return X, df_norm.columns.tolist()


# ============================================================
# DirectLiNGAM without bootstrapping
# ============================================================

def fit_direct_lingam(X, random_state=42):

    model = lingam.DirectLiNGAM(
        random_state=random_state
    )

    model.fit(X)

    # IMPORTANT:
    #
    # LiNGAM stores:
    #
    #     adjacency[target, source]
    #
    # We transpose so our saved matrices use:
    #
    #     adjacency[source, target]
    #
    # matching the NOTEARS matrices.

    W = model.adjacency_matrix_.T.copy()

    np.fill_diagonal(W, 0.0)

    return W


# ============================================================
# DirectLiNGAM with bootstrapping
# ============================================================

def fit_direct_lingam_bootstrap(
    X,
    n_bootstrap=100,
    bootstrap_threshold=0.5,
    random_state=42
):

    rng = np.random.RandomState(random_state)

    n_samples, n_features = X.shape

    weight_sums = np.zeros(
        (n_features, n_features)
    )

    edge_counts = np.zeros(
        (n_features, n_features)
    )

    successful_runs = 0

    for b in range(n_bootstrap):

        # Sample subjects with replacement
        idx = rng.choice(
            n_samples,
            size=n_samples,
            replace=True
        )

        try:
            model = lingam.DirectLiNGAM(
                random_state=random_state + b
            )

            model.fit(X[idx])

            # Transpose to:
            # row = source
            # column = target
            adj = model.adjacency_matrix_.T.copy()

            np.fill_diagonal(adj, 0.0)

            # Sum weights only where an edge exists
            weight_sums += adj

            # Count how often each edge occurs
            edge_counts += (
                np.abs(adj) > 0
            ).astype(float)

            successful_runs += 1

        except Exception as e:

            print(
                f"  Bootstrap {b + 1} failed: {e}"
            )

    if successful_runs == 0:
        raise RuntimeError(
            "All DirectLiNGAM bootstrap runs failed."
        )

    # Mean weight conditional on the edge existing
    mean_weights = np.divide(
        weight_sums,
        edge_counts,
        out=np.zeros_like(weight_sums),
        where=edge_counts > 0
    )

    # Fraction of successful bootstrap runs
    # containing each edge
    edge_probs = (
        edge_counts / successful_runs
    )

    # Keep edges appearing in at least the
    # requested proportion of bootstrap runs
    W = np.where(
        edge_probs >= bootstrap_threshold,
        mean_weights,
        0.0
    )

    print(
        f"\nSuccessful bootstrap runs: "
        f"{successful_runs}/{n_bootstrap}"
    )

    return W, edge_probs


# ============================================================
# Print edges
# ============================================================

def print_edges(W, col_names):

    rows, cols = np.where(W != 0)

    print(f"\nNumber of edges: {len(rows)}")

    if len(rows) == 0:
        print("  None")
        return

    print("\nLearned edges:")

    for r, c in zip(rows, cols):

        print(
            f"  {col_names[r]} -> "
            f"{col_names[c]}: "
            f"{W[r, c]:.4f}"
        )


# ============================================================
# Run DirectLiNGAM and save outputs
# ============================================================

def run_and_save(
    name,
    df,
    n_bootstrap=100,
    bootstrap_threshold=0.5
):

    print("\n" + "=" * 70)
    print(f"Running DirectLiNGAM on: {name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    df_norm = normalize_icv(
        df.copy(),
        name=name
    )

    X, col_names = preprocess(df_norm)

    print("\nDirectLiNGAM input:")
    print(f"  Samples:   {X.shape[0]}")
    print(f"  Variables: {X.shape[1]}")

    # ========================================================
    # 1. DirectLiNGAM WITHOUT bootstrapping
    # ========================================================

    print("\n" + "-" * 70)
    print("DirectLiNGAM without bootstrapping")
    print("-" * 70)

    W_single = fit_direct_lingam(
        X,
        random_state=42
    )

    single_df = pd.DataFrame(
        W_single,
        index=col_names,
        columns=col_names
    )

    single_path = (
        f"adjacency_matrix_directlingam_{name}.csv"
    )

    single_df.to_csv(single_path)

    print(f"\nSaved:")
    print(f"  {single_path}")

    print_edges(
        W_single,
        col_names
    )

    # ========================================================
    # 2. DirectLiNGAM WITH bootstrapping
    # ========================================================

    print("\n" + "-" * 70)
    print(
        f"DirectLiNGAM with "
        f"{n_bootstrap} bootstrap runs"
    )
    print("-" * 70)

    W_boot, edge_probs = fit_direct_lingam_bootstrap(
        X,
        n_bootstrap=n_bootstrap,
        bootstrap_threshold=bootstrap_threshold,
        random_state=42
    )

    boot_df = pd.DataFrame(
        W_boot,
        index=col_names,
        columns=col_names
    )

    boot_path = (
        f"adjacency_matrix_directlingam_"
        f"{name}_bootstrap.csv"
    )

    boot_df.to_csv(boot_path)

    print(f"\nSaved:")
    print(f"  {boot_path}")

    # Also save bootstrap edge probabilities.
    probability_df = pd.DataFrame(
        edge_probs,
        index=col_names,
        columns=col_names
    )

    probability_path = (
        f"edge_probabilities_directlingam_"
        f"{name}_bootstrap.csv"
    )

    probability_df.to_csv(
        probability_path
    )

    print(f"  {probability_path}")

    print_edges(
        W_boot,
        col_names
    )


# ============================================================
# Load datasets
# ============================================================

df_t1 = pd.read_csv(
    "T1_synthseg_vols_robust_no_parc.csv"
)

df_ppmi = pd.read_csv(
    "PPMI_33_feat.csv"
)

hcp_df = pd.read_csv(
    "synthseg_HCP.csv"
)


# ============================================================
# Dataset information
# ============================================================

print("\nDataset sizes:")
print(f"  PPMI subjects: {len(df_ppmi)}")
print(f"  T1 subjects:   {len(df_t1)}")
print(f"  HCP subjects:  {len(hcp_df)}")


# ============================================================
# Run DirectLiNGAM separately on all three datasets
# ============================================================

run_and_save(
    "PPMI",
    df_ppmi,
    n_bootstrap=100,
    bootstrap_threshold=0.5
)

run_and_save(
    "T1",
    df_t1,
    n_bootstrap=100,
    bootstrap_threshold=0.5
)

run_and_save(
    "HCP",
    hcp_df,
    n_bootstrap=100,
    bootstrap_threshold=0.5
)
