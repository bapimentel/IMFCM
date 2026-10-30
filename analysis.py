import numpy as np
from scipy.stats import ranksums, wilcoxon
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


METRICS = ['ARI', 'NMI', 'FR']
ALPHA = 0.05

SCATTER_AXES = ["$x$", "$y$", "$z$"]
SCATTER_MARKERS = ["o", "s", "^", "D", "v", "P"]
SCATTER_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
SCATTER_PANEL = 2.3
SCATTER_STYLE = {
    "font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 9, "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.6,
    "axes.edgecolor": "0.3", "axes.linewidth": 0.8, "font.family": "serif",
    "mathtext.fontset": "cm", "axes.facecolor": "white", "figure.facecolor": "white",
}


def FR_Index(U, ref):
    U = np.asarray(U, dtype=float)
    ref = np.asarray(ref).reshape(-1)

    if U.ndim != 2:
        raise ValueError(f'U must be a two-dimensional matrix, got shape={U.shape}.')

    if U.shape[0] != ref.size:
        if U.shape[1] == ref.size:
            U = U.T
        else:
            raise ValueError(f'U does not match the number of labels: U.shape={U.shape}, len(ref)={ref.size}.')

    classes, ref_encoded = np.unique(ref, return_inverse=True)
    nClass = U.shape[1]

    if len(classes) != nClass:
        raise ValueError(f'The reference has {len(classes)} classes but U has {nClass} clusters.')

    nObj = U.shape[0]
    nPairs = nObj * (nObj - 1) / 2.0

    if np.all(U >= -1e-12) and np.all(U.sum(axis=1) <= 1.0 + 1e-9):
        sum_all = nObj * np.sum((U - U.mean(axis=0)) ** 2)
        sum_same = 0.0
        pairs_same = 0.0
        for k in range(nClass):
            Uk = U[ref_encoded == k]
            nk = Uk.shape[0]
            if nk > 1:
                sum_same += nk * np.sum((Uk - Uk.mean(axis=0)) ** 2)
            pairs_same += nk * (nk - 1) / 2.0
        pairs_diff = nPairs - pairs_same
        total = (2.0 * pairs_diff + 2.0 * sum_same - sum_all) / nClass
    else:
        total = sum_pairs(U, ref_encoded, nClass)

    return 1.0 - total / nPairs


def sum_pairs(U, labels, nClass, block=256):
    nObj = U.shape[0]
    norms = np.sum(U ** 2, axis=1)
    total = 0.0
    for a in range(0, nObj, block):
        b = min(nObj, a + block)
        distp = norms[a:b, None] + norms[None, :] - 2.0 * (U[a:b] @ U.T)
        distq = 2.0 * (labels[a:b, None] != labels[None, :])
        diff = np.abs(distp - distq) / nClass
        upper = np.arange(nObj)[None, :] > np.arange(a, b)[:, None]
        total += diff[upper].sum()
    return total


def evaluate(L, ref, U):
    ari = adjusted_rand_score(L, ref)
    nmi = normalized_mutual_info_score(ref, L)
    fr = FR_Index(U, ref)
    return [ari, nmi, fr]


def holm(p_values):
    names = sorted(p_values, key=p_values.get)
    k = len(names)
    adjusted, accumulated = {}, 0.0
    for i, name in enumerate(names):
        accumulated = max(accumulated, min(1.0, (k - i) * p_values[name]))
        adjusted[name] = accumulated
    return adjusted


def rank_sum_tests(reference, others):
    p, z, sign = {}, {}, {}
    for name, values in others.items():
        result = ranksums(reference, values)
        p[name] = result.pvalue
        z[name] = result.statistic
        sign[name] = int(np.sign(np.median(reference) - np.median(values)))
    p_holm = holm(p)
    return {name: (p_holm[name], z[name], symbol(p_holm[name], sign[name])) for name in others}


def symbol(p, sign):
    if p >= ALPHA:
        return '~'
    return '+' if sign > 0 else '-'


def signed_rank_tests(ranks, method):
    rows = {}
    for name in ranks.columns:
        if name == method:
            continue
        d = ranks[method].to_numpy() - ranks[name].to_numpy()
        result = wilcoxon(d, zero_method='wilcox', correction=True, alternative='less', method='approx')
        nz = d[d != 0]
        w_minus = nz.size * (nz.size + 1) / 2.0 - result.statistic
        rows[name] = {'W+': result.statistic, 'W-': w_minus, 'n': nz.size, 'z': result.zstatistic,
                      'p': result.pvalue, 'r_rb': (w_minus - result.statistic) / (w_minus + result.statistic)}
    p_holm = holm({name: row['p'] for name, row in rows.items()})
    for name in rows:
        rows[name]['p_Holm'] = p_holm[name]
    return rows
