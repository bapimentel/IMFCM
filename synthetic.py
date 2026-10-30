import os

for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(var, '1')

import random
import sys
import time
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta, norm, ranksums, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from algorithms import M_GRID, execute, recommend_m
from analysis import (ALPHA, METRICS, SCATTER_AXES, SCATTER_COLORS, SCATTER_MARKERS, SCATTER_PANEL, SCATTER_STYLE,
                      evaluate, holm, rank_sum_tests)
from datasets import SCENARIO, SHAPE, SYNTHETIC, beta_sd, synthetic_dataset


SEED = 42
MC = 100
NREP = 100
ORDER = ['GK', 'FCM', 'FCM-D', 'FCM-S', 'MFCM', 'FCM-MFCM', 'FCM-WD', 'IMFCM']
RULES = {'m1': '1st peak dJ/dm', 'm2': '1st peak d2J/dm2', 'm3': '1st peak d3J/dm3'}
RESULTS = Path(__file__).resolve().parent / 'results' / 'synthetic'

# figure of the shape experiment: fixed colour per algorithm (in plotting order), markers as a colour-free encoding
ALGORITHM_STYLE = {
    'IMFCM': ('#2a78d6', 'o'), 'FCM-D': ('#eb6834', 's'), 'GK': ('#1baf7a', '^'), 'FCM-WD': ('#eda100', 'D'),
    'FCM-S': ('#e87ba4', 'v'), 'MFCM': ('#008300', 'P'), 'FCM-MFCM': ('#6250d6', 'X'), 'FCM': ('#e34948', '*'),
}
SHAPE_TEX = [r'$\mathcal{N}$', '5', '2', '1', '0.5', '0.2']
SHAPE_GRAYS = ['#b0afa8', '#98978f', '#7f7e78', '#66655f', '#4a4946', '#0b0b0b']

# fidelity of the estimated relevance (FCM has no variable weights)
WEIGHTED = ['GK', 'FCM-D', 'FCM-S', 'MFCM', 'FCM-MFCM', 'FCM-WD', 'IMFCM']
WEIGHTS = RESULTS / 'weights'


def monte_carlo(indexData, method, parM):
    rows = []
    for i in range(MC):
        dataset, ref, nClusters, dataName = synthetic_dataset(indexData)

        centersMC = np.zeros((NREP, nClusters))
        for c in range(NREP):
            centersMC[c] = random.sample(range(1, len(dataset)), nClusters)

        best = execute(NREP, dataset, centersMC, method, parM)
        rows.append(evaluate(best[1], ref, best[2]) + [best[0]])
    return pd.DataFrame(rows, columns=METRICS + ['J'])


def run_performance(task):
    method, indexData = task
    start = time.time()
    np.random.seed(SEED)
    random.seed(SEED)
    df = monte_carlo(indexData, method, 2)
    path = RESULTS / ('shape' if indexData in SHAPE else 'performance') / indexData / f'{method}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return f'{method} on {indexData}', time.time() - start


def run_sensitivity(indexData):
    start = time.time()
    np.random.seed(SEED)
    random.seed(SEED)
    for m in M_GRID:
        df = monte_carlo(indexData, 'IMFCM', m)
        path = RESULTS / 'sensitivity' / indexData / f'm_{m:.1f}.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return f'IMFCM on {indexData}, {len(M_GRID)} values of m', time.time() - start


def run_pool(function, tasks):
    with Pool(min(len(tasks), os.cpu_count()), maxtasksperchild=1) as pool:
        for k, (name, seconds) in enumerate(pool.imap_unordered(function, tasks), start=1):
            print(f'[{k}/{len(tasks)}] {name}: {seconds:.0f} s', flush=True)


def mean_sd(x):
    return f'{x.mean():.4f} ({x.std():.4f})'


def p_text(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def performance_tables(folder='performance', ids=SYNTHETIC):
    out = RESULTS / folder
    runs = {(d, a): pd.read_csv(out / d / f'{a}.csv') for d in ids for a in ORDER}
    competitors = ORDER[:-1]
    scenario = all(d in SCENARIO for d in ids)
    summary = []

    for metric in METRICS:
        table, tests = [], []
        wtl = {a: {'+': 0, '~': 0, '-': 0} for a in competitors}
        for d in ids:
            imfcm = runs[(d, 'IMFCM')][metric].to_numpy()
            res = rank_sum_tests(imfcm, {a: runs[(d, a)][metric].to_numpy() for a in competitors})
            row = {'Dataset': d, 'Scenario': SCENARIO[d]} if scenario else {'Dataset': d}
            test_row = dict(row)
            for a in ORDER:
                x = runs[(d, a)][metric]
                row[a] = mean_sd(x)
                record = {'Dataset': d, 'Algorithm': a, 'Metric': metric, 'Mean': x.mean(), 'SD': x.std()}
                if a != 'IMFCM':
                    p, z, s = res[a]
                    row[a] += s
                    test_row[a] = f'{p_text(p)} ({z:.2f}){s}'
                    wtl[a][s] += 1
                    record.update({'p_Holm': p, 'z': z, 'Result': s})
                summary.append(record)
            table.append(row)
            tests.append(test_row)
        table.append({'Dataset': 'W/T/L', **({'Scenario': ''} if scenario else {}),
                      **{a: f"{wtl[a]['+']}/{wtl[a]['~']}/{wtl[a]['-']}" for a in competitors}, 'IMFCM': '--'})
        pd.DataFrame(table).to_csv(out / f'{metric.lower()}.csv', index=False)
        pd.DataFrame(tests).to_csv(out / f'wilcoxon_{metric.lower()}.csv', index=False)

    pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)


def check_shape():
    # D10_0.2 draws the same random numbers as D10, so its replications must equal those of the performance step
    # (FR and J may differ in the last bits of the floating-point sums)
    if not (RESULTS / 'performance' / 'D10').exists():
        print('Check of D10_0.2 skipped: run the performance step first')
        return
    same = [np.allclose(pd.read_csv(RESULTS / 'shape' / 'D10_0.2' / f'{a}.csv'),
                        pd.read_csv(RESULTS / 'performance' / 'D10' / f'{a}.csv'), rtol=0, atol=1e-10) for a in ORDER]
    print(f'D10_0.2 equal to D10 (up to 1e-10) for {sum(same)} of {len(ORDER)} algorithms', flush=True)


def shape_density(level, x):
    # density of the modified components of D10_alpha without the class shift
    if level == 'N':
        return norm.pdf(x, 0.5, beta_sd(0.2))
    alpha = float(level)
    s = beta_sd(0.2) / beta_sd(alpha)
    u = (x - 0.5) / s + 0.5
    out = np.zeros_like(x)
    inside = (u > 0) & (u < 1)
    out[inside] = beta.pdf(u[inside], alpha, alpha) / s
    return out


def plot_shape():
    runs = {(d, a): pd.read_csv(RESULTS / 'shape' / d / f'{a}.csv') for d in SHAPE for a in ORDER}
    levels = [d.split('_')[1] for d in SHAPE]
    pos = np.arange(len(SHAPE))
    x = np.linspace(-1.5, 2.0, 4001)

    for with_fr in (True, False):
        with plt.rc_context(SCATTER_STYLE):
            if with_fr:
                fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.4))
            else:
                fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.75))
            axes = axes.ravel()

            for level, tex, gray in zip(levels, SHAPE_TEX, SHAPE_GRAYS):
                axes[0].plot(x, shape_density(level, x), color=gray, lw=1.3,
                             label=tex if level == 'N' else rf'$\alpha={tex}$')
            axes[0].set_ylim(0, 3.0)
            axes[0].set_xlim(-1.5, 1.9)
            axes[0].set_xticks([-1, 0, 1], ['$-1$', '$0$', '$1$'])
            axes[0].set_xlabel(r'$x$ (class 1)')
            axes[0].set_ylabel('density')
            axes[0].set_title('(a) same mean and SD', loc='left')
            axes[0].legend(frameon=False, fontsize=7, loc='upper left', ncol=1, handlelength=1.0,
                           handletextpad=0.4, borderaxespad=0.2)

            for ax, metric, letter in zip(axes[1:], METRICS if with_fr else METRICS[:2], 'bcd'):
                for a, (color, marker) in ALGORITHM_STYLE.items():
                    ax.plot(pos, [runs[(d, a)][metric].mean() for d in SHAPE], color=color, marker=marker,
                            ms=5 if a != 'FCM' else 6.5, mec='white', mew=0.6, lw=2.0 if a == 'IMFCM' else 1.0,
                            zorder=3 if a == 'IMFCM' else 2, label=a)
                ax.set_xticks(pos, SHAPE_TEX)
                ax.set_xlabel(r'shape parameter $\alpha$')
                ax.set_ylabel(f'mean {metric}')
                if metric != 'FR':
                    ax.set_ylim(0, 1)
                ax.set_title(f'({letter}) {metric}', loc='left')

            handles, labels = axes[1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=len(ORDER), frameon=False, fontsize=8,
                       handletextpad=0.3, columnspacing=0.9)
            fig.tight_layout(rect=(0, 0.07 if with_fr else 0.11, 1, 1))
            fig.savefig(RESULTS / 'shape' / ('shape_d10.png' if with_fr else 'shape_d10_noFR.png'), dpi=300)
            plt.close(fig)


def generator_sample(indexData, calls, seed):
    np.random.seed(seed)
    parts = [synthetic_dataset(indexData)[:2] for _ in range(calls)]
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def class_auc(X_train, y_train, X_test, y_test, cols):
    model = HistGradientBoostingClassifier(random_state=0).fit(X_train[:, cols], y_train)
    proba = model.predict_proba(X_test[:, cols])
    return np.array([roc_auc_score(y_test == k, proba[:, k]) for k in range(proba.shape[1])])


def run_truth(indexData):
    # true relevance from the generator only: homogeneity Var(X_j | i) / Var(X_j), and necessity, the drop of the
    # one-vs-rest AUC of class i when variable j is removed from a gradient-boosting classifier
    start = time.time()
    X_train, y_train = generator_sample(indexData, 500, 7)
    X_test, y_test = generator_sample(indexData, 500, 8)
    c, p = len(np.unique(y_train)), X_train.shape[1]
    S = np.array([X_train[y_train == k].var(axis=0) / X_train.var(axis=0) for k in range(c)])
    full = class_auc(X_train, y_train, X_test, y_test, list(range(p)))
    N = np.column_stack([full - class_auc(X_train, y_train, X_test, y_test, [h for h in range(p) if h != j])
                         for j in range(p)])
    path = WEIGHTS / 'truth' / f'{indexData}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'Class': np.arange(1, c + 1), 'AUC': full,
                  **{f'S{j + 1}': S[:, j] for j in range(p)}, **{f'N{j + 1}': N[:, j] for j in range(p)}}) \
        .to_csv(path, index=False)
    return f'relevance of {indexData}', time.time() - start


def load_truth(indexData):
    t = pd.read_csv(WEIGHTS / 'truth' / f'{indexData}.csv')
    p = sum(col.startswith('S') for col in t.columns)
    return t[[f'S{j + 1}' for j in range(p)]].to_numpy(), t[[f'N{j + 1}' for j in range(p)]].to_numpy()


def weights_of(best, method):
    # IMFCM returns [J, L, memb, count, U, P, A]; the other algorithms [J, L, memb, count, weights]
    return np.asarray(best[6] if method == 'IMFCM' else best[4], dtype=float)


def matched_weights(labels, ref, W):
    # weights of the cluster matched to each class by the Hungarian method, normalized to sum one per cluster
    c = W.shape[0]
    table = np.zeros((c, c))
    for k, l in zip(ref, labels):
        table[k, l] += 1
    rows, cols = linear_sum_assignment(-table)
    W = W[cols[np.argsort(rows)]]
    return W / W.sum(axis=1, keepdims=True)


def run_weights(task):
    method, indexData = task
    start = time.time()
    np.random.seed(SEED)
    random.seed(SEED)
    rows = []
    for i in range(MC):
        dataset, ref, nClusters, dataName = synthetic_dataset(indexData)
        centersMC = np.zeros((NREP, nClusters))
        for c in range(NREP):
            centersMC[c] = random.sample(range(1, len(dataset)), nClusters)
        best = execute(NREP, dataset, centersMC, method, 2)
        W = matched_weights(np.argmax(np.asarray(best[2]), axis=1), ref, weights_of(best, method))
        rows.append(evaluate(best[1], ref, best[2]) + [best[0]] + list(W.ravel()))
    cols = METRICS + ['J'] + [f'w{i + 1}{j + 1}' for i in range(W.shape[0]) for j in range(W.shape[1])]
    path = WEIGHTS / indexData / f'{method}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return f'{method} on {indexData}', time.time() - start


def relevance_correlations(indexData, method):
    S, N = load_truth(indexData)
    df = pd.read_csv(WEIGHTS / indexData / f'{method}.csv')
    W = df[[f'w{i + 1}{j + 1}' for i in range(S.shape[0]) for j in range(S.shape[1])]].to_numpy()
    # all-equal weights (FCM-MFCM with p = 2) carry no order of the variables: correlation set to zero
    nec = [0.0 if np.ptp(w) == 0 else spearmanr(w, N.ravel())[0] for w in W]
    hom = [0.0 if np.ptp(w) == 0 else spearmanr(w, -S.ravel())[0] for w in W]
    return pd.DataFrame({'necessity': nec, 'homogeneity': hom, 'ARI': df.ARI})


def relevance_tests(reference, others):
    # rank-sum tests of IMFCM against each competitor, Holm adjustment; the direction is given by the medians, or by
    # the means when the medians tie (the correlations take few distinct values)
    p, sign = {}, {}
    for name, x in others.items():
        p[name] = ranksums(reference, x).pvalue
        sign[name] = np.sign(np.median(reference) - np.median(x)) or np.sign(np.mean(reference) - np.mean(x))
    p_holm = holm(p)
    return {name: (p_holm[name], '~' if p_holm[name] >= ALPHA else ('+' if sign[name] > 0 else '-')) for name in others}


def relevance_table(values, algorithms, groups):
    rows, records = [], []
    wtl = {a: {'+': 0, '~': 0, '-': 0} for a in algorithms[:-1]}
    for d in SYNTHETIC:
        res = relevance_tests(values[(d, 'IMFCM')], {a: values[(d, a)] for a in algorithms[:-1]})
        row = {'Dataset': d}
        for a in algorithms:
            x = pd.Series(values[(d, a)])
            row[a] = mean_sd(x) + (res[a][1] if a != 'IMFCM' else '')
            record = {'Dataset': d, 'Algorithm': a, 'Mean': x.mean(), 'SD': x.std()}
            if a != 'IMFCM':
                wtl[a][res[a][1]] += 1
                record.update({'p_Holm': res[a][0], 'Result': res[a][1]})
            records.append(record)
        rows.append(row)
    for label, ids in groups:
        rows.append({'Dataset': label, **{a: f'{np.mean([values[(d, a)].mean() for d in ids]):.4f}' for a in algorithms}})
    rows.append({'Dataset': 'W/T/L', **{a: f"{wtl[a]['+']}/{wtl[a]['~']}/{wtl[a]['-']}" for a in algorithms[:-1]},
                 'IMFCM': '--'})
    return pd.DataFrame(rows), records


def weights_tables():
    corr = {(d, a): relevance_correlations(d, a) for d in SYNTHETIC for a in WEIGHTED}
    groups = [('Mean D1-D4', SYNTHETIC[:4]), ('Mean D5-D10', SYNTHETIC[4:]), ('Mean D1-D10', SYNTHETIC)]
    summary = []
    for reference in ['necessity', 'homogeneity']:
        table, records = relevance_table({k: v[reference].to_numpy() for k, v in corr.items()}, WEIGHTED, groups)
        table.to_csv(WEIGHTS / f'spearman_{reference}.csv', index=False)
        summary += [{'Table': f'spearman {reference}', **r} for r in records]
    # the same correlations on the replications in which each algorithm recovered the partition
    pd.DataFrame([{'Dataset': d, 'Algorithm': a, 'Replications': int((v.ARI >= 0.8).sum()),
                   'necessity': v[v.ARI >= 0.8].necessity.mean(), 'homogeneity': v[v.ARI >= 0.8].homogeneity.mean()}
                  for (d, a), v in corr.items()]).to_csv(WEIGHTS / 'spearman_ari_0.8.csv', index=False)
    pd.DataFrame(summary).to_csv(WEIGHTS / 'summary.csv', index=False)


def scatter_panel(ax, data, labels, x_idx, y_idx):
    names = SCATTER_AXES if data.shape[1] <= 3 else [f"$x_{{{i + 1}}}$" for i in range(data.shape[1])]
    for k, label in enumerate(np.unique(labels)):
        mask = labels == label
        ax.scatter(data[mask, x_idx], data[mask, y_idx], s=14,
                   marker=SCATTER_MARKERS[k % len(SCATTER_MARKERS)],
                   c=SCATTER_COLORS[k % len(SCATTER_COLORS)],
                   edgecolors="black", linewidths=0.3, alpha=0.75, label=f"Class {k + 1}")
    ax.set_xlabel(names[x_idx])
    ax.set_ylabel(names[y_idx])


def plot_scatter(synthetic, data_ref, save_path):
    synthetic = np.asarray(synthetic)
    data_ref = np.asarray(data_ref)
    pairs = list(combinations(range(synthetic.shape[1]), 2))
    cols = min(3, len(pairs))
    rows = -(-len(pairs) // cols)
    n_classes = len(np.unique(data_ref))
    legend_cols = n_classes if cols > 1 else min(2, n_classes)
    legend_height = 0.13 + 0.2 * -(-n_classes // legend_cols)

    with plt.rc_context(SCATTER_STYLE):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * SCATTER_PANEL, rows * SCATTER_PANEL + legend_height),
                                 squeeze=False)
        for ax, (x_idx, y_idx) in zip(axes.flat, pairs):
            scatter_panel(ax, synthetic, data_ref, x_idx, y_idx)
        for ax in axes.flat[len(pairs):]:
            ax.remove()

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=legend_cols, frameon=False,
                   bbox_to_anchor=(0.5, 1.0), markerscale=1.6, handletextpad=0.2, columnspacing=1.0)
        fig.tight_layout(rect=(0, 0, 1, 1 - legend_height / fig.get_figheight()))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)


def plot_scenario(samples, save_path):
    # 2D datasets share rows of three panels; each 3D dataset fills a row with its pairwise projections
    flat = [d for d in samples if samples[d][0].shape[1] == 2]
    rows = [[(d, (0, 1)) for d in flat[i:i + 3]] for i in range(0, len(flat), 3)]
    rows += [[(d, pair) for pair in combinations(range(3), 2)] for d in samples if samples[d][0].shape[1] == 3]
    n_classes = max(len(np.unique(ref)) for _, ref in samples.values())
    legend_height = 0.33
    row_height = SCATTER_PANEL + 0.3

    with plt.rc_context(SCATTER_STYLE):
        fig = plt.figure(figsize=(3 * SCATTER_PANEL, len(rows) * row_height + legend_height))
        grid = fig.add_gridspec(len(rows), 6)
        for r, row in enumerate(rows):
            start = 3 - len(row)
            panels = [d for d, _ in row]
            for k, (d, (x_idx, y_idx)) in enumerate(row):
                ax = fig.add_subplot(grid[r, start + 2 * k:start + 2 * k + 2])
                scatter_panel(ax, *samples[d], x_idx, y_idx)
                if k == panels.index(d) + panels.count(d) // 2:
                    ax.annotate(d, xy=(0.5, 0), xycoords=ax.xaxis.label, xytext=(0, -5),
                                textcoords='offset points', ha='center', va='top', fontsize=10)
                if len(np.unique(samples[d][1])) == n_classes:
                    handles, labels = ax.get_legend_handles_labels()

        fig.legend(handles, labels, loc="upper center", ncol=n_classes, frameon=False,
                   bbox_to_anchor=(0.5, 1.0), markerscale=1.6, handletextpad=0.2, columnspacing=1.0)
        fig.tight_layout(rect=(0, 0, 1, 1 - legend_height / fig.get_figheight()))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)


def plot_samples():
    samples = {}
    for d in SYNTHETIC:
        np.random.seed(SEED)
        dataset, ref, nClusters, dataName = synthetic_dataset(d)
        samples[d] = (np.asarray(dataset), np.asarray(ref))
        plot_scatter(dataset, ref, RESULTS / 'samples' / f'{d}.png')
    for scenario in sorted(set(SCENARIO.values())):
        plot_scenario({d: samples[d] for d in SYNTHETIC if SCENARIO[d] == scenario},
                      RESULTS / 'samples' / f'scenario_{scenario}.png')


def load_sensitivity(d):
    folder = RESULTS / 'sensitivity' / d
    return {m: pd.read_csv(folder / f'm_{m:.1f}.csv') for m in M_GRID}


def sensitivity_outputs():
    rows = []
    for d in SYNTHETIC:
        runs = load_sensitivity(d)
        df = pd.DataFrame([runs[m].mean() for m in M_GRID])
        df.insert(0, 'm', M_GRID)
        df.insert(0, 'Dataset', d)
        rows.append(df)

        fig = plt.figure(figsize=(12, 6))
        plt.plot(df["m"], df["ARI"], marker='o', label="ARI", color='dodgerblue')
        plt.plot(df["m"], df["NMI"], marker='s', label="NMI", color='darkorange')
        plt.plot(df["m"], df["FR"], marker='^', label="FR", color='green', linestyle='--')
        plt.xlabel("Fuzzifier value $m$", fontsize=14)
        plt.ylabel("Metric value", fontsize=14)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(RESULTS / 'sensitivity' / f'IMFCM_metrics_{d}.png', dpi=300)
        plt.close(fig)

    pd.concat(rows).to_csv(RESULTS / 'sensitivity' / 'means.csv', index=False)


def plot_derivatives(d, means, rec, path):
    max_ari_m = means.loc[means["ARI"].idxmax(), "m"]
    max_nmi_m = means.loc[means["NMI"].idxmax(), "m"]
    max_fr_m = means.loc[means["FR"].idxmax(), "m"]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    line_j, = ax1.plot(rec["grid"], rec["Jbar"], marker='o', linestyle='-', color='blue', label='Objective function $J$')
    ax1.set_xlabel("Fuzzifier value $m$", fontsize=14)
    ax1.set_ylabel("Objective function $J$", color='blue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.grid(True, linestyle='--', linewidth=0.5)

    ax2 = ax1.twinx()
    line_d1, = ax2.plot(rec["grid"], rec["d1"], marker='x', linestyle='--', color='red', label='First derivative $dJ/dm$')
    line_d2, = ax2.plot(rec["grid"], rec["d2"], marker='s', linestyle='-.', color='purple', label='Second derivative $d^2J/dm^2$')
    line_d3, = ax2.plot(rec["grid"], rec["d3"], marker='^', linestyle='-', color='brown', label='Third derivative $d^3J/dm^3$')
    ax2.set_ylabel("Derivatives of $J$", color='black', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='black', labelsize=12)

    line_ari = ax1.axvline(max_ari_m, color='green', linestyle=':', label=f'Max ARI at $m={max_ari_m}$')
    line_nmi = ax1.axvline(max_nmi_m, color='orange', linestyle=':', label=f'Max NMI at $m={max_nmi_m}$')
    line_fr = ax1.axvline(max_fr_m, color='deepskyblue', linestyle=':', label=f'Max FR at $m={max_fr_m}$')

    lines = [line_j, line_d1, line_d2, line_d3, line_ari, line_nmi, line_fr]
    ax1.legend(lines, [line.get_label() for line in lines], loc='upper right', fontsize=11)
    fig.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


def recommendation_outputs():
    out = RESULTS / 'recommendation'
    out.mkdir(parents=True, exist_ok=True)
    selected, derivatives, runs, means = [], [], {}, {}

    for d in SYNTHETIC:
        runs[d] = load_sensitivity(d)
        means[d] = pd.DataFrame([runs[d][m].mean() for m in M_GRID]).assign(m=M_GRID)
        rec = recommend_m(M_GRID, means[d]['J'].to_numpy())
        selected.append({'Dataset': d, 'Scenario': SCENARIO[d], **{RULES[r]: rec[r] for r in RULES}})
        derivatives.append(pd.DataFrame({'Dataset': d, 'm': rec['grid'], 'Jbar': rec['Jbar'],
                                         'dJ/dm': rec['d1'], 'd2J/dm2': rec['d2'], 'd3J/dm3': rec['d3']}))
        plot_derivatives(d, means[d], rec, out / f'IMFCM_m_all_{d}.png')

    selected = pd.DataFrame(selected)
    selected.to_csv(out / 'm_selected.csv', index=False)
    pd.concat(derivatives).to_csv(out / 'derivatives.csv', index=False)

    summary = []
    for metric in METRICS:
        table = []
        wtl = {r: {'+': 0, '~': 0, '-': 0} for r in RULES.values()}
        for _, sel in selected.iterrows():
            d = sel['Dataset']
            base = runs[d][2.0][metric]
            others = {r: runs[d][sel[r]][metric].to_numpy() for r in RULES.values() if sel[r] != 2.0}
            res = rank_sum_tests(base.to_numpy(), others) if others else {}
            row = {'Dataset': d, 'Scenario': SCENARIO[d], 'm = 2.0': mean_sd(base)}
            for r in RULES.values():
                x = runs[d][sel[r]][metric]
                row[r] = mean_sd(x)
                record = {'Dataset': d, 'Metric': metric, 'Rule': r, 'm': sel[r], 'Mean': x.mean(), 'SD': x.std()}
                if r in res:
                    p, z, s = res[r]
                    s = {'+': '-', '-': '+', '~': '~'}[s]
                    row[r] += s
                    wtl[r][s] += 1
                    record.update({'p_Holm': p, 'z': -z, 'Result': s})
                else:
                    wtl[r]['~'] += 1
                summary.append(record)
            table.append(row)
        table.append({'Dataset': 'W/T/L', 'Scenario': '', 'm = 2.0': '--',
                      **{r: f"{wtl[r]['+']}/{wtl[r]['~']}/{wtl[r]['-']}" for r in RULES.values()}})
        pd.DataFrame(table).to_csv(out / f'm_{metric.lower()}.csv', index=False)

    pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)


if __name__ == '__main__':
    steps = sys.argv[1:] or ['samples', 'performance', 'sensitivity', 'recommendation', 'shape', 'weights']
    for step in steps:
        start = time.time()
        if step == 'samples':
            plot_samples()
        elif step == 'performance':
            run_pool(run_performance, [(a, d) for d in SYNTHETIC for a in ORDER])
            performance_tables()
        elif step == 'sensitivity':
            run_pool(run_sensitivity, SYNTHETIC)
            sensitivity_outputs()
        elif step == 'recommendation':
            recommendation_outputs()
        elif step == 'shape':
            run_pool(run_performance, [(a, d) for d in SHAPE for a in ORDER])
            performance_tables('shape', SHAPE)
            check_shape()
            plot_shape()
        elif step == 'weights':
            run_pool(run_truth, SYNTHETIC)
            run_pool(run_weights, [(a, d) for d in SYNTHETIC for a in WEIGHTED])
            weights_tables()
        else:
            raise SystemExit(f'Unknown step: {step}. '
                             'Use samples, performance, sensitivity, recommendation, shape and/or weights.')
        print(f'{step} finished in {time.time() - start:.0f} s', flush=True)
