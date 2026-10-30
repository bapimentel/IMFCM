# IMFCM: Integrated Multivariate Fuzzy C-Means

## Overview

IMFCM is a multivariate fuzzy clustering algorithm that derives, from membership degrees computed per variable, a weight `a_ij` for each variable `j` in each cluster `i` and embeds these weights in the dissimilarity of its objective function. The paper also proposes dispersion-based indices to interpret partitions, clusters and variables, and an unsupervised rule that recommends the fuzzifier `m` from the objective function alone.

This repository reproduces every experiment reported in the paper:

| Part | Analysis | Command |
|---|---|---|
| Synthetic datasets | A. Clustering performance | `python synthetic.py performance` |
| Synthetic datasets | B. Shape of the within-cluster distributions at fixed dispersion | `python synthetic.py shape` |
| Synthetic datasets | C. Fidelity of the estimated relevance | `python synthetic.py weights` |
| Synthetic datasets | D. Sensitivity to the fuzzifier `m` | `python synthetic.py sensitivity` |
| Synthetic datasets | E. Automatic selection of the fuzzifier `m` | `python synthetic.py recommendation` |
| Real-world datasets | A. Clustering performance | `python real_data.py performance` |
| Real-world datasets | B. Case study: cluster interpretation on the Wine dataset | `python real_data.py wine` |
| Application | Recommended `m` + IMFCM + interpretation indices (breast cancer, TCGA) | `python application.py` |

## Algorithms

All eight algorithms are implemented in `algorithms.py` and use the names of the paper.

| Name | Algorithm | Function |
|---|---|---|
| **IMFCM** | Integrated Multivariate Fuzzy C-Means (proposed) | `IMFCM` |
| FCM | Fuzzy C-Means (Bezdek et al., 1984) | `FCM` |
| GK | Gustafson–Kessel, fuzzy covariance matrix (1979) | `GK` |
| FCM-S | FCM with an adaptive quadratic distance shared by all clusters (de Carvalho, 2006) | `FCM_S` |
| FCM-D | FCM with an adaptive quadratic distance per cluster (de Carvalho, 2006) | `FCM_D` |
| MFCM | Multivariate Fuzzy C-Means (Pimentel and de Souza, 2013) | `MFCM` |
| FCM-MFCM | Hybrid of FCM and MFCM, hybridization parameter 0.5 (Pimentel et al., 2022) | `FCM_MFCM` |
| FCM-WD | FCM with weighted membership and distance (Pimentel et al., 2022) | `FCM_WD` |

Every function receives the data matrix, the indices of the objects used as initial prototypes and `m`, and returns `[J, labels, memberships, iterations, weights]`, where `weights` is the `c x p` matrix of variable weights of the final solution:

| Algorithm | Weights |
|---|---|
| FCM | All equal to 1 (no variable weights) |
| GK | Diagonal of the norm-inducing matrix `det(F_i)^(1/p) F_i^-1` of each cluster (`gk_weights`) |
| FCM-S | The weights shared by all clusters, repeated in each row |
| FCM-D | The weights of each cluster |
| FCM-WD | The variable weights of each cluster (`W2`) |
| MFCM, FCM-MFCM | `compute_aij` applied to their multivariate memberships (same formula as IMFCM, without feedback into the dissimilarity) |

`IMFCM` returns `[J, labels, memberships, iterations, u_kij, prototypes, a_ij]`: the multivariate memberships `u_kij` (one `n x c` matrix per variable), the prototypes and the weights `a_ij`, which are the inputs of the interpretation indices.

The methodological contributions shared by the experiments and the application are also in `algorithms.py`:

* `compute_interpretation_indices(u, x, y, m)`: overall index `R = B/T`, `COR(j)`, `CTR(j)`, the relative contributions `T(i)`, `J(i)`, `B(i)` (keys `T_rel`, `J_rel`, `B_rel`), `COR(i,j)`, `CTR(i,j)` and `CE(i,j)`, computed with the weights `w_kij` of the IMFCM prototype update. `dispersions` returns `T_ij`, `J_ij`, `B_ij`, which satisfy `T_ij = J_ij + B_ij`.
* `recommend_m(grid, Jbar)`: first, second and third derivatives of the mean objective function by the weighted finite differences of the paper (`numpy.gradient` on the non-uniform grid) and the first peak of each derivative (`first_peak`; the global maximum when there is no interior peak). The recommended value is the first peak of the third derivative, `m3`.

## Requirements

* Python 3.12 (tested with 3.12.10)
* numpy 2.0.0, scipy 1.13.1, scikit-learn 1.5.0, pandas 2.3.3, matplotlib 3.9.0
* ucimlrepo 0.0.7, only needed to download a UCI dataset again if its CSV is removed from `data/uci`

The versions are pinned in `requirements.txt`.

## Installation

```bash
git clone https://github.com/bapimentel/IMFCM.git
```

```bash
cd IMFCM
```

```bash
python -m pip install -r requirements.txt
```

All commands below are run from the root of the repository. The experiments use all available CPU cores (`multiprocessing`), one single-threaded process per task.

## Files

| File | Content |
|---|---|
| `algorithms.py` | The eight algorithms, `execute` (best of `nRep` initializations by the objective function), interpretation indices, recommendation of `m` |
| `datasets.py` | Generators of the ten synthetic datasets and of the D10_α family, loading and preprocessing of the 28 UCI datasets |
| `analysis.py` | Fuzzy Rand index, ARI/NMI/FR evaluation, Wilcoxon tests with Holm adjustment, plotting style |
| `synthetic.py` | Synthetic experiments (performance, shape of the distributions, fidelity of the weights, sensitivity to `m`, selection of `m`) |
| `real_data.py` | UCI experiments (performance, Wine case study) |
| `application.py` | Breast cancer application |
| `data/uci/D1.csv`, ..., `data/uci/D28.csv` | Snapshot of the 28 UCI datasets |
| `data/brca/brca_tcga.csv` | Snapshot of the TCGA breast cancer data used in the application |

## Experiments on synthetic datasets

`python synthetic.py` draws the sample figures and runs the five steps below. The steps are described in the order of the paper and run in the order `performance`, `sensitivity`, `recommendation`, `shape`, `weights`. The ten datasets (`D1`, ..., `D10`, numbered as in Tables 1–3 of the paper; `D1`–`D4` are Scenario 1, `D5`–`D6` Scenario 2 and `D7`–`D10` Scenario 3) are generated by `datasets.synthetic_dataset`. Each experiment is a Monte Carlo study with 100 replications: in each replication a new sample is generated, the algorithm is run from 100 random initializations, and the run with the lowest objective function is evaluated by ARI, NMI and FR.

```bash
python synthetic.py samples
```

Draws one sample of each dataset (seed 42). Outputs in `results/synthetic/samples/`:

| File | Paper |
|---|---|
| `<id>.png` | The sample of one dataset (its pairwise projections if it is three-dimensional) |
| `scenario_<k>.png` | The samples of the datasets of Scenario `k` (figures of Scenarios 1–3). Two-dimensional datasets share the first row; each three-dimensional dataset fills a row with its three pairwise projections. Each dataset is labelled by its id under its panel(s) |

### A. Clustering performance

```bash
python synthetic.py performance
```

Runs the eight algorithms with `m = 2` on the ten datasets. Outputs in `results/synthetic/performance/`:

| File | Paper |
|---|---|
| `<id>/<algorithm>.csv` | ARI, NMI, FR and `J` of each of the 100 replications |
| `ari.csv`, `nmi.csv`, `fr.csv` | Tables of ARI, NMI and FR: mean (standard deviation), result of the test against IMFCM and W/T/L |
| `wilcoxon_ari.csv`, `wilcoxon_nmi.csv`, `wilcoxon_fr.csv` | Wilcoxon tables: Holm-adjusted p-value (standardized rank-sum statistic `z`) |
| `summary.csv` | The same values without rounding |

Test symbols: `+` IMFCM significantly better, `-` IMFCM significantly worse, `~` no significant difference (two-sided Wilcoxon rank-sum test, Holm adjustment over the seven comparisons of each dataset and index, `alpha = 0.05`).

### B. Shape of the within-cluster distributions at fixed dispersion

```bash
python synthetic.py shape
```

Runs the eight algorithms with `m = 2` on the six datasets of the family D10_α (`datasets.SHAPE`: `D10_N`, `D10_5`, `D10_2`, `D10_1`, `D10_0.5`, `D10_0.2`) with the Monte Carlo protocol of step A. In D10, variables `x` and `z` of classes 1 and 2 follow a U-shaped Beta(0.2, 0.2) distribution. In D10_α they follow Beta(α, α) rescaled to the mean and standard deviation of Beta(0.2, 0.2) (`datasets.shape_component`); `N` is the Gaussian limit with the same mean and standard deviation. Variable `y` and class 3 are those of D10. Only the shape of the four components changes, from Gaussian to flatter unimodal (α = 5 and 2), uniform (α = 1) and U-shaped (α = 0.5, a rescaled arcsine, and α = 0.2), so their within-class dispersions are the same at all levels. `D10_0.2` draws the same random numbers as `D10`; when the files of step A exist, the step checks that its replications are identical to those of `D10`. Outputs in `results/synthetic/shape/`:

| File | Paper |
|---|---|
| `<id>/<algorithm>.csv` | ARI, NMI, FR and `J` of each of the 100 replications |
| `ari.csv`, `nmi.csv`, `fr.csv` | Tables of ARI, NMI and FR on D10_α: mean (standard deviation), result of the test against IMFCM and W/T/L |
| `wilcoxon_ari.csv`, `wilcoxon_nmi.csv`, `wilcoxon_fr.csv` | Wilcoxon tables: Holm-adjusted p-value (standardized rank-sum statistic `z`) |
| `summary.csv` | The same values without rounding |
| `shape_d10.png` | Density of the modified components at each level and mean ARI, NMI and FR (figure of the shape experiment) |
| `shape_d10_noFR.png` | The same figure without FR |

Test symbols as in step A. The step takes about 2 minutes on 24 cores (about 31 minutes of CPU time).

### C. Fidelity of the estimated relevance

```bash
python synthetic.py weights
```

Compares the variable weights of the algorithms (table in the section *Algorithms*; FCM has none) with the true relevance of the variables, computed from the generators without any clustering algorithm:

* **homogeneity** of variable `j` in class `i`, `h_ij = Var(X_j | class i) / Var(X_j)`;
* **necessity**, `nu_ij`, the decrease of the area under the ROC curve of class `i` against the other classes when variable `j` is removed from a gradient-boosting classifier (`HistGradientBoostingClassifier`, `random_state=0`) trained on one sample of the generator and evaluated on another (500 draws of the dataset each, seeds 7 and 8).

The seven algorithms with weights are run with `m = 2` on D1-D10 with the Monte Carlo protocol of step A, which reproduces its replications. In each replication the clusters of the best solution are matched to the classes by the Hungarian method, the weights are normalized to sum one in each cluster, and the Spearman correlation between the weights and `nu_ij`, and between the weights and `-h_ij`, is computed over all classes and variables. With `p = 2`, the weights of FCM-MFCM are all equal and their correlation is set to zero.

Outputs in `results/synthetic/weights/`:

| File | Paper |
|---|---|
| `truth/<id>.csv` | AUC of the full classifier, `h_ij` (`S`) and `nu_ij` (`N`) of each class and variable |
| `<id>/<algorithm>.csv` | ARI, NMI, FR, `J` and the matched normalized weights `w_ij` of each replication |
| `spearman_necessity.csv`, `spearman_homogeneity.csv` | Tables of the correlations with necessity and homogeneity: mean (standard deviation), result of the test against IMFCM, means of D1-D4, D5-D10 and D1-D10, and W/T/L |
| `spearman_ari_0.8.csv` | Mean correlations on the replications in which each algorithm attained ARI >= 0.8 |
| `summary.csv` | The values of the tables without rounding, with the Holm-adjusted p-values |

Test symbols as in step A. The direction of a significant difference is given by the medians, or by the means when the medians tie, because the correlations take few distinct values. The step takes about 4 minutes on 24 cores.

### D. Sensitivity to the fuzzifier

```bash
python synthetic.py sensitivity
```

Runs IMFCM for the 17 values of `m` in `algorithms.M_GRID` (1.1, 1.3, 1.5, 1.7, 2.0, 2.2, 2.5, 2.7, 3.0, 3.2, 3.5, 3.7, 4.0, 4.3, 4.5, 4.7, 5.0), with the Monte Carlo protocol above for each value. For each dataset, the values of `m` are run in increasing order within one random stream. Outputs in `results/synthetic/sensitivity/`:

| File | Paper |
|---|---|
| `<id>/m_<m>.csv` | ARI, NMI, FR and `J` of each replication for each `m` |
| `means.csv` | Mean ARI, NMI, FR and `J` for each dataset and `m` |
| `IMFCM_metrics_<id>.png` | Mean ARI, NMI and FR as functions of `m` (figure of the sensitivity analysis, datasets D1 and D10 in the paper) |

### E. Automatic selection of the fuzzifier

```bash
python synthetic.py recommendation
```

Uses only the files of step D. For each dataset, `recommend_m` computes the derivatives of the mean objective function `Jbar(m)` and the first peak of each derivative; the replications of the selected values of `m` are then compared with those of `m = 2.0`. Outputs in `results/synthetic/recommendation/`:

| File | Paper |
|---|---|
| `m_selected.csv` | Values of `m` selected by the first peak of `dJ/dm`, `d2J/dm2` and `d3J/dm3` |
| `derivatives.csv` | `Jbar` and its three derivatives for each dataset and `m` |
| `m_ari.csv`, `m_nmi.csv`, `m_fr.csv` | ARI, NMI and FR with `m = 2.0` and with the selected values, test against `m = 2.0` and W/T/L |
| `summary.csv` | The same values without rounding |
| `IMFCM_m_all_<id>.png` | `Jbar` and its derivatives, with the values of `m` that maximize ARI, NMI and FR (figure of the automatic selection, datasets D1 and D10 in the paper) |

Test symbols: `+` the rule is significantly better than `m = 2.0`, `-` significantly worse, `~` no significant difference; no symbol when the rule selects `m = 2.0`, which counts as a tie in W/T/L. The two-sided rank-sum tests are Holm-adjusted over the rules of each dataset and index whose selected value differs from 2.0.

## Experiments on real-world datasets

`python real_data.py` runs the two steps below in order.

The 28 UCI datasets are identified as in the paper (`D1`, ..., `D28`, numbered independently of the synthetic datasets; `datasets.REAL` gives the UCI identifier, name and number of classes of each one). They are the subset reported in the paper, selected among the 44 datasets of the original runs with the same data, so tests on them are exploratory. Preprocessing (`datasets.preprocess`): constant attributes are removed; numeric attributes with more than two distinct values are standardized (z-score); the first `min(5, p_Q)` principal components of the numeric block are retained; categorical attributes are encoded as one 0/1 column (binary attributes, missing value coded as -1) or one indicator column per category (missing value: all zeros) and are not standardized nor included in the PCA. Instances with missing numeric values are removed when the CSV is created.

### A. Clustering performance

```bash
python real_data.py performance
```

Runs the eight algorithms with `m = 2` and 200 random initializations on each dataset; the run with the lowest objective function is evaluated. Each run starts from the same seed, so on each dataset the eight algorithms draw the same 200 sets of initial prototypes. The files in `runs/` are the results of the full run of the original experiment scripts (`experiment2_datasets_parallel.py` of the earlier `Clustering-py` repository) under condition C03: standard scaler, five principal components, no whitening.

Outputs in `results/real/`:

| File | Paper |
|---|---|
| `runs/<id>_<algorithm>.csv`, `scores.csv` | ARI, NMI, FR and `J` of each algorithm and dataset |
| `ari.csv`, `nmi.csv`, `fr.csv` | Tables of the 28 UCI datasets: value (within-dataset rank) and, in the last row, mean (mean rank) |
| `ranks_ari.csv`, `ranks_nmi.csv`, `ranks_fr.csv` | Within-dataset ranks (1 = best, ties receive the average rank) |
| `wilcoxon.csv` | One-sided paired Wilcoxon signed-rank tests of IMFCM against each competitor on the ranks (alternative: IMFCM attains lower ranks; zero differences discarded; normal approximation with tie and continuity corrections), Holm-adjusted p-values over the seven comparisons and rank-biserial correlation `r_rb` |
| `uniform_partitions.csv` | Datasets in which the FR of each algorithm equals, to four decimal places, the FR of the uniform partition (all memberships equal to `1/c`) |

### B. Case study: cluster interpretation on the Wine dataset

```bash
python real_data.py wine
```

Runs IMFCM on Wine (`D5`) exactly as in step A (same random stream, same 200 initializations, so the same partition) and computes the interpretation indices of the best partition. Outputs in `results/real/wine/`:

| File | Paper |
|---|---|
| `pc_attributes.csv` | Correlations between the 13 standardized attributes and the five principal components, and prototypes mapped back to the standardized attributes |
| `clusters.csv` | `T(i)`, `J(i)`, `B(i)` |
| `variables.csv` | Explained variance of each component, `COR(j)`, `CTR(j)` |
| `COR_ij.csv`, `CTR_ij.csv`, `CE_ij.csv` | Cluster- and component-level indices |
| `prototypes_pc.csv`, `weights_a_ij.csv` | Prototypes in the space of the components and weights `a_ij` |
| `clusters_vs_cultivar.csv` | Composition of the clusters by cultivar (reference only; not used by the indices) |
| `summary.json` | ARI, NMI, FR, `J`, `R` and the variance explained by the five components |

Cluster numbers follow the order of the prototypes in the best run; `clusters_vs_cultivar.csv` gives their correspondence with the cultivars. In the paper the clusters are numbered by cultivar (`C1`: cultivar 1, `C2`: cultivar 3, `C3`: cultivar 2), which corresponds to `C3`, `C2` and `C1` of these files.

## Application

```bash
python application.py
```

Breast cancer data of the TCGA PanCancer Atlas (cBioPortal study `brca_tcga_pan_can_atlas_2018`; IHC status from study `brca_tcga`): expression of ESR1, PGR, ERBB2, GRB7 and MKI67 in the 945 primary tumors with PAM50 subtype luminal A, luminal B, HER2-enriched or basal-like. Expression is transformed by `log2(x + 1)` and standardized. The workflow combines the three components of the paper:

1. **Automatic recommendation of `m`**: IMFCM is run for every value of `M_GRID` on 30 bootstrap resamples (20 initializations each), `J` is averaged over the resamples and `recommend_m` selects `m` by the first peak of the third derivative.
2. **IMFCM clustering** with the recommended `m` and `k = 4` (100 initializations); the seven competitors are run with `m = 2` under the same protocol.
3. **Interpretation**: weights `a_ij` and indices `CTR(i,j)`, `COR(i,j)`, `T(i)`, `J(i)`, `B(i)` of `compute_interpretation_indices`, validated against the subtypes, the IHC groups, Cohen's `d` and within-subtype homogeneity, 200 bootstrap resamples and the neighbouring values of `m`.

The subtypes and the IHC status are used only after the fit. Outputs in `results/application/`:

| File | Paper |
|---|---|
| `tables/data_medians_by_subtype.csv` | Genes and median expression by subtype |
| `tables/m_selection.csv`, `figures/brca_m_selection.png` | `Jbar` and its derivatives, selected `m` |
| `tables/clusters_vs_ihc.csv` | IMFCM clusters versus IHC groups |
| `tables/interpretation_*.csv` | `a_ij` (`A`), `CTR(i,j)`, `COR(i,j)`, cluster and variable indices, standardized profiles |
| `figures/brca_profiles.png`, `figures/brca_imfcm_weights.png` | Profiles by subtype and by cluster; `a_ij` and `CTR(i,j)` |
| `tables/rank_agreement.csv`, `tables/true_cohen_d.csv`, `tables/true_homogeneity.csv`, `figures/brca_interpretation_coherence.png` | Coherence of the indices with the data |
| `tables/bootstrap_*.csv` | Bootstrap confidence intervals, stability of the most and least relevant genes, Jaccard similarity |
| `tables/interpretation_across_m.csv` | Agreement and indices for the values of `m` around the selected one |
| `tables/membership_vs_subtype.csv`, `figures/brca_memberships.png` | Membership degrees of concordant and discordant tumors |
| `tables/agreement.csv` | ARI, NMI, matched accuracy and FR of IMFCM and the seven competitors |
| `summary.json` | Selected values of `m` and the numbers quoted in the text |

## Data

* `data/uci/D1.csv`, ..., `data/uci/D28.csv`: one file per dataset with the numeric attributes, the categorical attributes (prefix `cat_`) and the reference class (`class`), after removing instances with missing numeric values. The files were created by `datasets.download_uci` from the UCI Machine Learning Repository (`ucimlrepo`, licence CC BY 4.0) and are identical to the data used in the paper. If a file is removed, it is downloaded again on first use.
* `data/brca/brca_tcga.csv`: expression (RSEM) of the five genes, PAM50 subtype and IHC fields of the 981 TCGA tumors with expression and subtype, created by `application.download` from the cBioPortal API. If the file is removed, it is downloaded again on first use.

## Reproducibility

* **Seeds.** Every task (one algorithm on one synthetic or UCI dataset, or the whole sweep of `m` on one synthetic dataset) starts with `numpy.random.seed(42)` and `random.seed(42)`. Synthetic samples use the `numpy` generator; initial prototypes are drawn with `random.sample(range(1, n), c)`. GK also draws its initial memberships from the `numpy` generator, so its synthetic samples differ from those of the other algorithms, and the Monte Carlo replications are not paired across algorithms. The UCI datasets are fixed, so all algorithms draw the same initial prototypes. The application uses `numpy.random.default_rng(42)` (bootstrap resample `b` uses seed `42 + b`).
* **Repetitions.** Synthetic: 100 replications x 100 initializations. UCI: 200 initializations. Application: 100 initializations for the final fits, 20 for each bootstrap fit, 30 bootstrap resamples for `m` and 200 for the stability analysis.
* **Stopping.** `|J(t) - J(t+1)| < 1e-4` or 100 iterations.
* **NMI.** `sklearn.metrics.normalized_mutual_info_score` with its default arithmetic-mean normalization, as in all published results.
* **Standard deviations.** Sample standard deviations (`n - 1` denominator). The published tables of the recommendation of `m` and the IMFCM column of the synthetic performance tables used the population standard deviation, which is 0.5% smaller (for example 0.0582 instead of 0.0585).
* **Threads.** BLAS libraries are limited to one thread per process.

### Agreement with the published results

REPLACE_AGREEMENT
