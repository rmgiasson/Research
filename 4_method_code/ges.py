import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from causallearn.search.ScoreBased.GES import ges
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

    # Remove missing observations
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

    # ICV is used ONLY for normalization.
    # It is NOT included as a node in GES.
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

    X = scaler.fit_transform(
        df_norm.values
    )

    X = np.asarray(
        X,
        dtype=np.float64
    )

    return X, df_norm.columns.tolist()
# ============================================================
# Extract complete GES CPDAG
# ============================================================

def extract_ges_graph(G, n):
    """
    Convert the causal-learn GES graph into one adjacency matrix.

    Encoding:

        0 = no edge

        1 = directed edge
            cpdag[i, j] = 1 means i -> j

        2 = undirected edge
            cpdag[i, j] = 2 AND
            cpdag[j, i] = 2 means i -- j

    Therefore:
        row    = source
        column = target

    for directed edges.
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

        else:
            print(
                "WARNING: Unexpected GES edge type:",
                edge
            )

    return cpdag


# ============================================================
# Print complete CPDAG
# ============================================================

def print_graph(cpdag, col_names):

    n = len(col_names)

    directed_edges = []
    undirected_edges = []

    for i in range(n):
        for j in range(n):

            # Directed edge i -> j
            if cpdag[i, j] == 1:
                directed_edges.append(
                    (i, j)
                )

            # Undirected edge i -- j
            #
            # Only inspect i < j so the symmetric
            # representation isn't counted twice.
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
# Run GES and save complete CPDAG
# ============================================================

def run_and_save(name, df):

    print("\n" + "=" * 70)
    print(f"Running GES on: {name}")
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

    print("\nGES input:")
    print(
        f"  Samples:   "
        f"{X.shape[0]}"
    )
    print(
        f"  Variables: "
        f"{X.shape[1]}"
    )

    # --------------------------------------------------------
    # Run GES
    # --------------------------------------------------------

    print("\nRunning GES...")

    result = ges(X)

    G = result["G"]

    # --------------------------------------------------------
    # Extract complete CPDAG
    # --------------------------------------------------------

    cpdag = extract_ges_graph(
        G,
        len(col_names)
    )

    # --------------------------------------------------------
    # Save complete CPDAG
    # --------------------------------------------------------

    cpdag_df = pd.DataFrame(
        cpdag,
        index=col_names,
        columns=col_names
    )

    output_path = (
        f"adjacency_matrix_ges_{name}.csv"
    )

    cpdag_df.to_csv(
        output_path
    )

    print("\nSaved:")
    print(
        f"  {output_path}"
    )

    # --------------------------------------------------------
    # Print graph
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
    "/blue/neurology-dept/JOSH/_UF_DBS/"
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
    f"  PPMI subjects: "
    f"{len(df_ppmi)}"
)
print(
    f"  T1 subjects:   "
    f"{len(df_t1)}"
)
print(
    f"  HCP subjects:  "
    f"{len(hcp_df)}"
)


# ============================================================
# Run GES separately on all three datasets
# ============================================================

run_and_save(
    "PPMI",
    df_ppmi
)

run_and_save(
    "T1",
    df_t1
)

run_and_save(
    "HCP",
    hcp_df
)
