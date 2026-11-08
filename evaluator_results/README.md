# Five-node synthetic benchmark results

- `individual_results.csv`: all 80 evaluations (four methods, Gaussian/Laplace noise, seeds 1-10), with all evaluator metrics and saved algorithm runtimes.
- `summary.csv`: means and sample standard deviations (ddof=1) over ten seeds per method/noise group. Values retain full CSV precision.
- `README.md`: this overview, readable summary, and interpretation notes.

## Summary: mean ± sample SD

All scores are on a 0-1 scale. Exact skeleton and Markov-equivalence recovery are binary per-run indicators; their means are recovery fractions. Their SDs describe variation across seeds, not standard errors or confidence intervals.

| Noise | Method | Output | Precision | Recall | F1 | Exact skeleton | Compelled recovery | MEC recovery | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| gaussian | DirectLiNGAM | DAG | 0.704 ± 0.087 | 1.000 ± 0.000 | 0.824 ± 0.061 | 0.000 ± 0.000 | 0.500 ± 0.408 | 0.000 ± 0.000 | 0.023 ± 0.001 |
| gaussian | GES | CPDAG | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.044 ± 0.030 |
| gaussian | NOTEARS | DAG | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.459 ± 0.041 |
| gaussian | PC-Stable | CPDAG | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.100 ± 0.316 | 0.100 ± 0.316 | 0.007 ± 0.001 |
| laplace | DirectLiNGAM | DAG | 0.983 ± 0.053 | 1.000 ± 0.000 | 0.991 ± 0.029 | 0.900 ± 0.316 | 1.000 ± 0.000 | 0.900 ± 0.316 | 0.022 ± 0.000 |
| laplace | GES | CPDAG | 0.967 ± 0.070 | 1.000 ± 0.000 | 0.982 ± 0.038 | 0.800 ± 0.422 | 1.000 ± 0.000 | 0.800 ± 0.422 | 0.024 ± 0.026 |
| laplace | NOTEARS | DAG | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.561 ± 0.120 |
| laplace | PC-Stable | CPDAG | 0.983 ± 0.053 | 1.000 ± 0.000 | 0.991 ± 0.029 | 0.900 ± 0.316 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.006 ± 0.000 |

## Direction-related interpretation

DirectLiNGAM and NOTEARS supply fully directed DAGs. PC-Stable and GES supply CPDAG representations; undirected edges remain undirected and are not forced into DAGs. `estimated_undirected_edges_total` records the actual number per run.

- `compelled_edge_recovery_rate`: correctly recovered compelled directions / all true compelled edges. Missing, reversed, or undirected compelled edges do not count as correct. The true graph has two compelled edges.
- `compelled_orientation_accuracy_given_present`: correctly recovered compelled directions / true compelled edges present in the estimated skeleton. This excludes missing adjacencies from the denominator; both mean and SD appear in `summary.csv`.
- The three reversible true edges are not required to follow the arbitrary generating-DAG direction. Leaving them undirected is appropriate for CPDAG outputs. The column reporting matches to the generating DAG is descriptive only and is not included as an accuracy metric in the summary.
- For fully directed outputs, Markov-equivalence recovery compares the CPDAG derived from the estimated DAG to the true CPDAG. For outputs containing undirected edges, the evaluator compares the supplied partially directed graph literally to the true CPDAG; it does not complete general PDAGs first.
- All 80 Markov-equivalence verdicts are available, and no evaluated graph has a directed cycle. `n_markov_verdicts_available` makes the denominator explicit.

## Sources and runtime

These are existing common-evaluator results from `synthetic_common_evaluation_outputs`; no discovery algorithm or evaluation was rerun to make this folder. The batch's `laplace_seed1` results previously matched Praneeth's reference to absolute numerical tolerance 1e-12. This does not establish complete evaluator validation.

DirectLiNGAM/PC-Stable/GES matrices and their saved runtimes come from `5node_common_benchmark/common_5node_benchmark_outputs`. Corrected NOTEARS matrices and runtimes come from `common_5node_method_outputs`: the saved absolute coefficient threshold is 0.1. DirectLiNGAM has no additional coefficient threshold; evaluator numerical edge tolerance is 1e-8. CPDAG edge codes are adapted without losing undirected edges.

Runtimes are the original per-fit algorithm measurements, in seconds, matched by method/noise/seed; they are not evaluator runtimes. Timing was recorded in existing runs and was not remeasured under a new controlled timing experiment, so comparisons are descriptive. All 80 saved runtimes are present.

To regenerate this folder from existing local results, run `python summarize_synthetic_evaluation.py` from the workspace root. MRI pathway results are excluded.
