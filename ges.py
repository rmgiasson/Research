import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from causallearn.search.ScoreBased.GES import ges

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
    df = df[["total intracranial"] + brain_cols].copy()
    icv = df["total intracranial"].values.reshape(-1, 1)
    residuals = np.zeros((len(df), len(brain_cols)))
    for i, col in enumerate(brain_cols):
        reg = LinearRegression()
        reg.fit(icv, df[col].values)
        residuals[:, i] = df[col].values - reg.predict(icv)
    scaler = StandardScaler()
    return scaler.fit_transform(residuals)

# --------------------------------------------------
# Load data
# --------------------------------------------------
df_t1 = pd.read_csv("T1_synthseg_vols_robust_no_parc.csv")
print(f"Loaded {len(df_t1)} subjects")

X = preprocess(df_t1)
print(f"Preprocessed shape: {X.shape}")

# --------------------------------------------------
# Run GES
# --------------------------------------------------
print("Running GES...")
result = ges(X)

# result['G'] is a GeneralGraph object
# Extract adjacency matrix: G.graph[i,j]=1 and G.graph[j,i]=-1 means i->j
G = result['G']
n = len(brain_cols)
adj = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        # i->j: G.graph[j,i]=1 and G.graph[i,j]=-1
        if G.graph[j, i] == 1 and G.graph[i, j] == -1:
            adj[j, i] = 1.0  # edge from i to j, stored as adj[j,i] (parent i -> child j)

# Save labeled CSV
adj_df = pd.DataFrame(adj, index=brain_cols, columns=brain_cols)
adj_df.to_csv("ges_adjacency_T1.csv")
print(f"Saved: ges_adjacency_T1.csv")
print(f"Total directed edges: {int(adj.sum())}")

# Print edges
print("\nDirected edges (col -> row):")
found = False
for i in range(n):
    for j in range(n):
        if adj[i, j] == 1:
            print(f"  {brain_cols[j]} -> {brain_cols[i]}")
            found = True
if not found:
    print("  None found")
