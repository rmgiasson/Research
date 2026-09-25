import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt
from sklearn.preprocessing import StandardScaler
import networkx as nx


# ============================================================
# Linear NOTEARS
# ============================================================

def run_notears(
    X,
    lambda1=0.01,
    max_iter=100,
    h_tol=1e-8,
    rho_max=1e16,
    w_threshold=0.2,
    verbose=True
):
    """
    Linear NOTEARS with L1 regularization.

    Solves:

        min_W  (1 / 2n) ||X - XW||_F^2 + lambda1 * ||W||_1

        subject to:

        h(W) = tr(exp(W o W)) - d = 0

    W[i, j] represents the edge:

        variable i -> variable j
    """

    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")

    n, d = X.shape

    if n < 2:
        raise ValueError("NOTEARS requires at least 2 observations.")

    # --------------------------------------------------------
    # Acyclicity constraint
    # h(W) = tr(exp(W o W)) - d
    # --------------------------------------------------------
    def _h(W):
        E = slin.expm(W * W)
        h = np.trace(E) - d
        return h, E

    # --------------------------------------------------------
    # L1 variable splitting
    #
    # W = W_pos - W_neg
    # --------------------------------------------------------
    def _adj(w):
        W_pos = w[:d * d]
        W_neg = w[d * d:]

        return (W_pos - W_neg).reshape(d, d)

    # --------------------------------------------------------
    # Augmented-Lagrangian objective and gradient
    # --------------------------------------------------------
    def _func(w):
        W = _adj(w)

        h, E = _h(W)

        # Residuals
        R = X - X @ W

        # Least-squares loss
        loss = 0.5 / n * np.sum(R ** 2)

        # Gradient of least-squares loss
        G_loss = -(1.0 / n) * X.T @ R

        # Gradient of acyclicity constraint
        G_h = 2.0 * W * E.T

        # Augmented-Lagrangian objective
        obj = (
            loss
            + lambda1 * np.sum(w)
            + 0.5 * rho * h * h
            + alpha * h
        )

        # Gradient with respect to W
        G_smooth = G_loss + (rho * h + alpha) * G_h

        # Convert gradient to split-variable representation
        grad = np.concatenate(
            (
                G_smooth + lambda1,
                -G_smooth + lambda1
            ),
            axis=None
        )

        return obj, grad

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------
    w_est = np.zeros(2 * d * d)

    rho = 1.0
    alpha = 0.0
    h = np.inf

    # --------------------------------------------------------
    # Bounds
    #
    # Diagonal is fixed to zero.
    # Split variables are otherwise >= 0.
    # --------------------------------------------------------
    bounds = []

    for _ in range(2):
        for i in range(d):
            for j in range(d):
                if i == j:
                    bounds.append((0.0, 0.0))
                else:
                    bounds.append((0.0, None))

    # --------------------------------------------------------
    # Dual ascent
    # --------------------------------------------------------
    for iteration in range(max_iter):

        w_new = None
        h_new = None

        while rho < rho_max:

            solution = sopt.minimize(
                _func,
                w_est,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds
            )

            if not solution.success and verbose:
                print(
                    "  L-BFGS-B warning:",
                    solution.message
                )

            w_new = solution.x

            W_new = _adj(w_new)

            h_new, _ = _h(W_new)

            # Increase rho and solve again if the DAG
            # constraint did not improve enough.
            if h_new > 0.25 * h:
                rho *= 10.0
            else:
                break

        if w_new is None:
            raise RuntimeError(
                "NOTEARS optimization failed before producing a solution."
            )

        w_est = w_new
        h = h_new

        # Dual variable update
        alpha += rho * h

        if verbose:
            W_current = _adj(w_est)

            R = X - X @ W_current
            loss = 0.5 / n * np.sum(R ** 2)

            print(
                f"Iteration {iteration + 1:02d} | "
                f"Loss: {loss:.6f} | "
                f"h(W): {h:.3e} | "
                f"rho: {rho:.1e} | "
                f"alpha: {alpha:.3e}"
            )

        if h <= h_tol:
            if verbose:
                print("DAG constraint satisfied.")
            break

        if rho >= rho_max:
            if verbose:
                print("rho reached rho_max.")
            break

    # --------------------------------------------------------
    # Recover adjacency matrix
    # --------------------------------------------------------
    W_raw = _adj(w_est)

    np.fill_diagonal(W_raw, 0.0)

    # Threshold weak edges
    W_est = W_raw.copy()
    W_est[np.abs(W_est) < w_threshold] = 0.0

    return W_est, W_raw


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
    "right ventral DC",
]

cols_to_keep = ["total intracranial"] + brain_cols


# ============================================================
# ICV normalization
# ============================================================

def normalize_icv(df, name="dataset"):
    """
    Normalize each regional brain volume by total intracranial
    volume (ICV).
    """

    # Explicitly verify that all required columns exist.
    missing_cols = [
        col for col in cols_to_keep
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"{name} is missing required columns:\n"
            + "\n".join(missing_cols)
        )

    # Only retain columns actually used by NOTEARS.
    df = df[cols_to_keep].copy()

    before = len(df)

    # Remove observations missing any required volume.
    df = df.dropna()

    after_nan = len(df)

    # ICV must be positive.
    df = df[df["total intracranial"] > 0].copy()

    after_icv = len(df)

    print(f"\n{name} preprocessing:")
    print(f"  Original subjects:          {before}")
    print(f"  After NaN removal:          {after_nan}")
    print(f"  After invalid ICV removal:  {after_icv}")

    if after_icv == 0:
        raise ValueError(
            f"No valid subjects remain in {name}."
        )

    icv = df["total intracranial"]

    df_norm = df[brain_cols].div(icv, axis=0)

    # Defensive check
    if not np.isfinite(df_norm.values).all():
        raise ValueError(
            f"{name} contains non-finite values after ICV normalization."
        )

    return df_norm


# ============================================================
# Standardization
# ============================================================

def preprocess(df_norm):
    """
    Standardize ICV-normalized regional volumes before NOTEARS.
    """

    scaler = StandardScaler()

    X = scaler.fit_transform(df_norm.values)

    X = np.asarray(X, dtype=np.float64)

    return X, df_norm.columns.tolist()


# ============================================================
# Run NOTEARS and save graph
# ============================================================

def run_and_save(name, df):

    print("\n" + "=" * 70)
    print(f"Running NOTEARS on: {name}")
    print("=" * 70)

    # ICV normalization
    df_norm = normalize_icv(
        df.copy(),
        name=name
    )

    # Standardization
    X, col_names = preprocess(df_norm)

    print(f"\nNOTEARS input:")
    print(f"  Samples:   {X.shape[0]}")
    print(f"  Variables: {X.shape[1]}")

    # --------------------------------------------------------
    # Run NOTEARS
    # --------------------------------------------------------
    W_est, W_raw = run_notears(
        X,
        lambda1=0.01,
        max_iter=100,
        h_tol=1e-8,
        rho_max=1e16,
        w_threshold=0.2,
        verbose=True
    )

    # --------------------------------------------------------
    # Save raw matrix
    # --------------------------------------------------------
    W_raw_df = pd.DataFrame(
        W_raw,
        index=col_names,
        columns=col_names
    )

    raw_path = f"adjacency_matrix_{name}_raw.csv"

    W_raw_df.to_csv(raw_path)

    print(f"\nSaved raw matrix:")
    print(f"  {raw_path}")

    # --------------------------------------------------------
    # Save thresholded DAG
    # --------------------------------------------------------
    W_df = pd.DataFrame(
        W_est,
        index=col_names,
        columns=col_names
    )

    threshold_path = f"adjacency_matrix_{name}.csv"

    W_df.to_csv(threshold_path)

    print(f"\nSaved thresholded matrix:")
    print(f"  {threshold_path}")

    # --------------------------------------------------------
    # Extract edges
    # --------------------------------------------------------
    rows, cols = np.where(W_est != 0)

    print(f"\nNumber of thresholded edges: {len(rows)}")

    print("\nLearned edges:")

    if len(rows) == 0:
        print("  None")

    for r, c in zip(rows, cols):
        print(
            f"  {col_names[r]} -> "
            f"{col_names[c]}: "
            f"{W_est[r, c]:.4f}"
        )

    # --------------------------------------------------------
    # Verify DAG
    # --------------------------------------------------------
    G = nx.DiGraph()

    G.add_nodes_from(col_names)

    for r, c in zip(rows, cols):
        G.add_edge(
            col_names[r],
            col_names[c],
            weight=W_est[r, c]
        )

    is_dag = nx.is_directed_acyclic_graph(G)

    print(f"\nIs thresholded graph a DAG? {is_dag}")

    if not is_dag:
        cycles = list(nx.simple_cycles(G))

        print("WARNING: thresholded graph contains cycles.")

        if cycles:
            print("Example cycle:")
            print(" -> ".join(cycles[0] + [cycles[0][0]]))

    return W_est


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
# Run baseline NOTEARS separately on each dataset
# ============================================================

# PPMI-only DAG
run_and_save(
    "PPMI",
    df_ppmi
)

# T1-only DAG
run_and_save(
    "T1",
    df_t1
)

# HCP-only DAG
run_and_save(
    "HCP",
    hcp_df
)
