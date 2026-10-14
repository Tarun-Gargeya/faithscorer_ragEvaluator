# Results

Thresholds are tuned on the constructed dev split and frozen for all test sets. For RAGTruth, a second balanced accuracy uses thresholds tuned on RAGTruth's train split (`ragtruth_qa_dev`, disjoint questions), for every scorer alike.


## test  (n=2880, 1462 faithful / 1418 unfaithful)

| System | Bal. acc | F1 (unfaithful) | AUROC | ms / example |
|---|---|---|---|---|
| nli | 0.723 | 0.715 | 0.799 | 25.9 |
| nli_base | 0.762 | 0.741 | 0.844 | 118.3 |
| alignscore | 0.815 | 0.825 | 0.868 | 80.7 |
| minicheck | 0.881 | 0.879 | 0.941 | 265.4 |
| faithscorer | 0.959 | 0.958 | 0.989 | 27.1 |
| **faithscorer_sent** | 0.961 | 0.960 | 0.989 | 26.3 |
| faithscorer_mean | 0.959 | 0.958 | 0.982 | 25.1 |

Per-condition accuracy (label: 1 = faithful, 0 = unfaithful):

| Condition | label | n | nli | nli_base | alignscore | minicheck | faithscorer | faithscorer_sent | faithscorer_mean |
|---|---|---|---|---|---|---|---|---|---|
| cf_contradicted | 0 | 357 | 0.52 | 0.66 | 0.84 | 0.75 | 0.90 | 0.89 | 0.91 |
| cf_followed | 1 | 361 | 0.75 | 0.81 | 0.66 | 0.85 | 0.95 | 0.96 | 0.94 |
| clean | 1 | 369 | 0.75 | 0.87 | 0.76 | 0.92 | 0.98 | 0.98 | 0.98 |
| incomplete | 0 | 357 | 0.68 | 0.68 | 0.85 | 0.82 | 0.92 | 0.93 | 0.94 |
| irrelevant | 0 | 343 | 0.88 | 0.80 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 |
| noisy | 1 | 369 | 0.59 | 0.77 | 0.60 | 0.89 | 0.96 | 0.97 | 0.96 |
| supported_extra | 1 | 363 | 0.87 | 0.89 | 0.95 | 0.90 | 0.98 | 0.98 | 0.97 |
| unsupported_extra | 0 | 361 | 0.74 | 0.62 | 0.87 | 0.93 | 0.98 | 0.98 | 0.98 |

### LLM judge `gptoss20b` on its 200-example subset

| System | Bal. acc | F1 (unfaithful) | κ with judge |
|---|---|---|---|
| gptoss20b (judge) | 0.880 | 0.886 | — |
| nli | 0.740 | 0.732 | 0.364 |
| nli_base | 0.795 | 0.792 | 0.472 |
| alignscore | 0.835 | 0.845 | 0.564 |
| minicheck | 0.885 | 0.882 | 0.652 |
| faithscorer | 0.965 | 0.964 | 0.751 |
| faithscorer_sent | 0.965 | 0.964 | 0.751 |
| faithscorer_mean | 0.960 | 0.960 | 0.760 |

Escalation (FaithScorer + judge on least-confident cases): 0% → 0.965, 20% → 0.950, 40% → 0.940, 60% → 0.930, 80% → 0.900, 100% → 0.880

## ragtruth_qa  (n=900, 740 faithful / 160 unfaithful)

| System | Bal. acc | F1 (unfaithful) | AUROC | ms / example | Bal. acc (RAGTruth-dev thr.) |
|---|---|---|---|---|---|
| nli | 0.530 | 0.217 | 0.654 | 26.8 | 0.634 |
| nli_base | 0.487 | 0.085 | 0.564 | 39.3 | 0.576 |
| alignscore | 0.616 | 0.368 | 0.774 | 268.8 | 0.727 |
| minicheck | 0.566 | 0.287 | 0.656 | 95.4 | 0.643 |
| faithscorer | 0.610 | 0.357 | 0.676 | 21.2 | 0.618 |
| **faithscorer_sent** | 0.685 | 0.414 | 0.823 | 48.4 | 0.748 |
| faithscorer_mean | 0.692 | 0.437 | 0.744 | 48.4 | 0.690 |

Per-condition accuracy (label: 1 = faithful, 0 = unfaithful):

| Condition | label | n | nli | nli_base | alignscore | minicheck | faithscorer | faithscorer_sent | faithscorer_mean |
|---|---|---|---|---|---|---|---|---|---|
| Evident Baseless Info | 0 | 95 | 0.21 | 0.06 | 0.35 | 0.28 | 0.54 | 0.96 | 0.84 |
| Evident Baseless Info+Evident Conflict | 0 | 3 | 0.33 | 0.00 | 0.33 | 0.00 | 1.00 | 1.00 | 0.67 |
| Evident Baseless Info+Evident Conflict+Subtle Baseless Info | 0 | 1 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| Evident Baseless Info+Subtle Baseless Info | 0 | 10 | 0.30 | 0.20 | 0.70 | 0.40 | 0.60 | 1.00 | 0.90 |
| Evident Conflict | 0 | 22 | 0.09 | 0.00 | 0.14 | 0.18 | 0.32 | 0.55 | 0.36 |
| Subtle Baseless Info | 0 | 29 | 0.21 | 0.07 | 0.38 | 0.38 | 0.41 | 0.93 | 0.66 |
| faithful | 1 | 740 | 0.86 | 0.91 | 0.88 | 0.84 | 0.73 | 0.47 | 0.65 |

### LLM judge `gptoss120b` on its 600-example subset

| System | Bal. acc | F1 (unfaithful) | κ with judge |
|---|---|---|---|
| gptoss120b (judge) | 0.797 | 0.604 | — |
| nli | 0.527 | 0.203 | 0.126 |
| nli_base | 0.479 | 0.063 | 0.064 |
| alignscore | 0.595 | 0.335 | 0.200 |
| minicheck | 0.573 | 0.302 | 0.209 |
| faithscorer | 0.587 | 0.338 | 0.199 |
| faithscorer_sent | 0.688 | 0.429 | 0.218 |
| faithscorer_mean | 0.674 | 0.433 | 0.289 |

Escalation (FaithScorer + judge on least-confident cases): 0% → 0.747, 20% → 0.797, 40% → 0.799, 60% → 0.796, 80% → 0.800, 100% → 0.797
