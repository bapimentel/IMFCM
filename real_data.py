import os

for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(var, '1')

import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from algorithms import ALGORITHMS, compute_interpretation_indices, execute
from analysis import METRICS, FR_Index, evaluate, signed_rank_tests
from datasets import REAL, SEED, UCI_DIR, read_uci, real_dataset, remove_constant_columns, scale_non_binary


NREP = 200
RESULTS = Path(__file__).resolve().parent / 'results' / 'real'


def run_once(indexData, method):
    dataset, ref, nClusters, dataName = real_dataset(indexData)

    centersMC = np.zeros((NREP, nClusters))
    for c in range(NREP):
        centersMC[c] = random.sample(range(1, len(dataset)), nClusters)

    best = execute(NREP, dataset, centersMC, method)
    return best, dataset, ref


def run_performance(task):
    method, indexData = task
    start = time.time()
    np.random.seed(SEED)
    random.seed(SEED)
    best, dataset, ref = run_once(indexData, method)
    row = [indexData, REAL[indexData][1], method] + evaluate(best[1], ref, best[2]) + [best[0]]
    path = RESULTS / 'runs' / f'{indexData}_{method}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=['Dataset', 'Name', 'Algorithm'] + METRICS + ['J']).to_csv(path, index=False)
    return f'{method} on {indexData}', time.time() - start


def performance_tables():
    scores = pd.concat([pd.read_csv(RESULTS / 'runs' / f'{d}_{a}.csv') for d in REAL for a in ALGORITHMS])
    scores.to_csv(RESULTS / 'scores.csv', index=False)

    uniform = {}
    for d in REAL:
        dataset, ref, nClusters, dataName = real_dataset(d)
        uniform[d] = FR_Index(np.full((len(ref), nClusters), 1.0 / nClusters), ref)

    tests = []
    for metric in METRICS:
        values = scores.pivot(index='Dataset', columns='Algorithm', values=metric).reindex(index=list(REAL), columns=ALGORITHMS)
        ranks = values.rank(axis=1, ascending=False, method='average')
        table = pd.DataFrame({a: [f'{v:.4f} ({r:g})' for v, r in zip(values[a], ranks[a])] for a in ALGORITHMS},
                             index=values.index)
        table.loc['Mean'] = [f'{values[a].mean():.4f} ({ranks[a].mean():.2f})' for a in ALGORITHMS]
        table.insert(0, 'Name', [REAL[d][1] for d in REAL] + [''])
        table.to_csv(RESULTS / f'{metric.lower()}.csv', index_label='Dataset')
        ranks.to_csv(RESULTS / f'ranks_{metric.lower()}.csv', index_label='Dataset')

        for name, row in signed_rank_tests(ranks, 'IMFCM').items():
            tests.append({'Metric': metric, 'Algorithm': name, **row})

    pd.DataFrame(tests).to_csv(RESULTS / 'wilcoxon.csv', index=False)

    fr = scores.pivot(index='Dataset', columns='Algorithm', values='FR').reindex(index=list(REAL), columns=ALGORITHMS)
    near = pd.DataFrame({a: [round(fr.loc[d, a], 4) == round(uniform[d], 4) for d in REAL] for a in ALGORITHMS},
                        index=list(REAL))
    near.insert(0, 'FR uniform', [uniform[d] for d in REAL])
    near.loc['Count'] = [''] + [int(near[a].sum()) for a in ALGORITHMS]
    near.to_csv(RESULTS / 'uniform_partitions.csv', index_label='Dataset')


def wine_case_study():
    out = RESULTS / 'wine'
    out.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    random.seed(SEED)
    best, X, ref = run_once('D5', 'IMFCM')
    J, L, memb, count, U, P, A = best
    ind = compute_interpretation_indices(U, X, P, 2)
    ari, nmi, fr = evaluate(L, ref, memb)

    cont, cat_df, _, names = read_uci('D5')
    Z = scale_non_binary(remove_constant_columns(cont))
    pca = PCA(n_components=X.shape[1], whiten=False, random_state=SEED).fit(Z)
    pcs = [f'PC{j + 1}' for j in range(X.shape[1])]
    clusters = [f'C{i + 1}' for i in range(P.shape[0])]

    corr = np.corrcoef(Z.T, X.T)[:Z.shape[1], Z.shape[1]:]
    prototypes = P @ pca.components_ + pca.mean_
    attributes = pd.DataFrame(corr, index=names, columns=pcs)
    attributes[clusters] = prototypes.T
    attributes.to_csv(out / 'pc_attributes.csv', index_label='Attribute', float_format='%.4f')

    pd.DataFrame({'Var.': pca.explained_variance_ratio_, 'COR(j)': ind['COR_j'], 'CTR(j)': ind['CTR_j']},
                 index=pcs).to_csv(out / 'variables.csv', index_label='Component', float_format='%.4f')
    pd.DataFrame({'T(i)': ind['T_rel'], 'J(i)': ind['J_rel'], 'B(i)': ind['B_rel']},
                 index=clusters).to_csv(out / 'clusters.csv', index_label='Cluster', float_format='%.4f')
    for key, name in (('COR_ij', 'COR_ij'), ('CTR_ij', 'CTR_ij'), ('CE_ij', 'CE_ij')):
        pd.DataFrame(ind[key].T, index=pcs, columns=clusters).to_csv(out / f'{name}.csv', index_label='Component',
                                                                     float_format='%.4f')
    pd.DataFrame(P, index=clusters, columns=pcs).to_csv(out / 'prototypes_pc.csv', index_label='Cluster',
                                                        float_format='%.4f')
    pd.DataFrame(A, index=clusters, columns=pcs).to_csv(out / 'weights_a_ij.csv', index_label='Cluster',
                                                        float_format='%.4f')

    cultivar = pd.read_csv(UCI_DIR / 'D5.csv')['class'].to_numpy()
    pd.crosstab(pd.Series([clusters[l] for l in L], name='Cluster'),
                pd.Series([f'Cultivar {c}' for c in cultivar], name='Class')).to_csv(out / 'clusters_vs_cultivar.csv')

    summary = {'ARI': ari, 'NMI': nmi, 'FR': fr, 'J': J, 'R': ind['R'],
               'explained_variance': float(pca.explained_variance_ratio_.sum())}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    steps = sys.argv[1:] or ['performance', 'wine']
    for step in steps:
        start = time.time()
        if step == 'performance':
            size = {}
            for d in REAL:
                dataset, ref, nClusters, dataName = real_dataset(d)
                size[d] = dataset.shape[0] * dataset.shape[1] ** 3
            tasks = sorted([(a, d) for d in REAL for a in ALGORITHMS],
                           key=lambda t: -size[t[1]] * (1000 if t[0] == 'IMFCM' else 1))
            with Pool(os.cpu_count(), maxtasksperchild=1) as pool:
                for k, (name, seconds) in enumerate(pool.imap_unordered(run_performance, tasks), start=1):
                    print(f'[{k}/{len(tasks)}] {name}: {seconds:.0f} s', flush=True)
            performance_tables()
        elif step == 'wine':
            wine_case_study()
        else:
            raise SystemExit(f'Unknown step: {step}. Use performance and/or wine.')
        print(f'{step} finished in {time.time() - start:.0f} s', flush=True)
