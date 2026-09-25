import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.graph.Endpoint import Endpoint

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

    missing_cols = [
        col for col in cols_to_keep
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"{name} is missing required columns:\n"
            + "\n".join(missing_cols)
        )

    df = df[cols_to_keep].copy()

    before = len(df)

    # Remove rows containing missing values
    df = df.dropna()

    after_nan = len(df)

    # Remove invalid ICV values
    df = df[
        df["total intracranial"] > 0
    ].copy()

    after_icv = len(df)

    print(f"\n{name} preprocessing:")
    print(f"  Original subjects:         {before}")
    print(f"  After NaN removal:         {after_nan}")
    print(f"  After invalid ICV removal: {after_icv}")

    if after_icv == 0:
        raise ValueError(
            f"No valid subjects remain in {name}."
        )

    # --------------------------------------------------------
    # Normalize each brain volume by ICV.
    #
    # ICV is NOT included as a node in PC-Stable.
    # --------------------------------------------------------

    icv = df["total intracranial"]

    df_norm = df[brain_cols].div(
        icv,
        axis=0
    )

    if not np.isfinite(
        df_norm.values
    ).all():
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

    X = scaler.fit_transform(
        df_norm.values
    )

    X = np.asarray(
        X,
        dtype=np.float64
    )

    return X, df_norm.columns.tolist()


# ============================================================
# Extract complete PC-Stable CPDAG
# ============================================================

def extract_pc_graph(G, n):
    """
    Convert the causal-learn PC-Stable graph into one matrix.

    Encoding:

        0 = no edge

        1 = directed edge
            cpdag[i, j] = 1 means i -> j

        2 = undirected edge
            cpdag[i, j] = 2 AND
            cpdag[j, i] = 2 means i -- j

    Therefore for directed edges:

        row    = source
        column = target
    """

    cpdag = np.zeros(
        (n, n),
        dtype=int
    )

    node_map = G.get_node_map()

    for edge in G.get_graph_edges():

        node1 = edge.get_node1()
        node2 = edge.get_node2()

        i = node_map[node1]
        j = node_map[node2]

        endpoint1 = edge.get_endpoint1()
        endpoint2 = edge.get_endpoint2()

        # ----------------------------------------------------
        # node1 -> node2
        # ----------------------------------------------------

        if (
            endpoint1 == Endpoint.TAIL
            and endpoint2 == Endpoint.ARROW
        ):
            cpdag[i, j] = 1

        # ----------------------------------------------------
        # node1 <- node2
        # ----------------------------------------------------

        elif (
            endpoint1 == Endpoint.ARROW
            and endpoint2 == Endpoint.TAIL
        ):
            cpdag[j, i] = 1

        # ----------------------------------------------------
        # node1 -- node2
        # ----------------------------------------------------

        elif (
            endpoint1 == Endpoint.TAIL
            and endpoint2 == Endpoint.TAIL
        ):
            cpdag[i, j] = 2
            cpdag[j, i] = 2

        # ----------------------------------------------------
        # Bidirected edge: node1 <-> node2
        #
        # This should not normally be part of an ordinary
        # PC CPDAG. Print a warning rather than silently
        # misrepresenting it as directed or undirected.
        # ----------------------------------------------------

        elif (
            endpoint1 == Endpoint.ARROW
            and endpoint2 == Endpoint.ARROW
        ):
            print(
                "WARNING: Bidirected edge encountered:",
                f"{node1} <-> {node2}"
            )

        else:
            print(
                "WARNING: Unexpected PC edge type:",
                edge
            )

    return cpdag


# ============================================================
# Print graph
# ============================================================

def print_graph(cpdag, col_names):

    n = len(col_names)

    directed_edges = []
    undirected_edges = []

    for i in range(n):
        for j in range(n):

            # Directed i -> j
            if cpdag[i, j] == 1:
                directed_edges.append(
                    (i, j)
                )

            # Undirected i -- j
            #
            # Only count once because it is represented
            # symmetrically in the matrix.
            elif (
                i < j
                and cpdag[i, j] == 2
                and cpdag[j, i] == 2
            ):
                undirected_edges.append(
                    (i, j)
                )

    print(
        f"\nDirected edges: "
        f"{len(directed_edges)}"
    )

    for i, j in directed_edges:
        print(
            f"  {col_names[i]} -> "
            f"{col_names[j]}"
        )

    print(
        f"\nUndirected edges: "
        f"{len(undirected_edges)}"
    )

    for i, j in undirected_edges:
        print(
            f"  {col_names[i]} -- "
            f"{col_names[j]}"
        )

    print(
        f"\nTotal CPDAG edges: "
        f"{len(directed_edges) + len(undirected_edges)}"
    )


# ============================================================
# Run PC-Stable and save output
# ============================================================

def run_and_save(
    name,
    df,
    alpha=0.05
):

    print("\n" + "=" * 70)
    print(f"Running PC-Stable on: {name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    df_norm = normalize_icv(
        df.copy(),
        name=name
    )

    X, col_names = preprocess(
        df_norm
    )

    print("\nPC-Stable input:")
    print(
        f"  Samples:   {X.shape[0]}"
    )
    print(
        f"  Variables: {X.shape[1]}"
    )
    print(
        f"  Alpha:     {alpha}"
    )
    print(
        "  CI test:   Fisher's Z"
    )

    # --------------------------------------------------------
    # Run PC-Stable
    # --------------------------------------------------------

    print("\nRunning PC-Stable...")

    cg = pc(
        X,
        alpha=alpha,
        indep_test=fisherz,
        stable=True,
        uc_rule=0,
        uc_priority=2,
        verbose=False,
        show_progress=True
    )

    # --------------------------------------------------------
    # Extract complete CPDAG
    # --------------------------------------------------------

    cpdag = extract_pc_graph(
        cg.G,
        len(col_names)
    )

    # --------------------------------------------------------
    # Save adjacency matrix
    # --------------------------------------------------------

    cpdag_df = pd.DataFrame(
        cpdag,
        index=col_names,
        columns=col_names
    )

    output_path = (
        f"adjacency_matrix_pcstable_{name}.csv"
    )

    cpdag_df.to_csv(
        output_path
    )

    print("\nSaved:")
    print(
        f"  {output_path}"
    )

    # --------------------------------------------------------
    # Print learned graph
    # --------------------------------------------------------

    print_graph(
        cpdag,
        col_names
    )

    return cpdag


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

print(
    f"  PPMI subjects: {len(df_ppmi)}"
)

print(
    f"  T1 subjects:   {len(df_t1)}"
)

print(
    f"  HCP subjects:  {len(hcp_df)}"
)


# ============================================================
# Run PC-Stable separately on all three datasets
# ============================================================

run_and_save(
    "PPMI",
    df_ppmi,
    alpha=0.05
)

run_and_save(
    "T1",
    df_t1,
    alpha=0.05
)

run_and_save(
    "HCP",
    hcp_df,
    alpha=0.05
)
