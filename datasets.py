from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import arcsine, gumbel_r
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SEED = 42
UCI_DIR = Path(__file__).resolve().parent / 'data' / 'uci'

SYNTHETIC = [f'D{k}' for k in range(1, 11)]
SCENARIO = {'D1': 1, 'D2': 1, 'D3': 1, 'D4': 1, 'D5': 2, 'D6': 2, 'D7': 3, 'D8': 3, 'D9': 3, 'D10': 3}
# D10 with its four Beta(0.2, 0.2) components replaced by Beta(alpha, alpha) at the same mean and SD; N = Gaussian limit
SHAPE = ['D10_N', 'D10_5', 'D10_2', 'D10_1', 'D10_0.5', 'D10_0.2']

REAL = {
    'D1': (52, 'Ionosphere', 2),
    'D2': (70, "Monk's Problem", 2),
    'D3': (151, 'Sonar', 2),
    'D4': (17, 'WDBC', 2),
    'D5': (109, 'Wine', 3),
    'D6': (111, 'Zoo', 7),
    'D7': (42, 'Glass', 6),
    'D8': (277, 'Thoracic Surgery', 2),
    'D9': (19, 'Car Evaluation', 4),
    'D10': (95, 'SPECT Heart', 2),
    'D11': (22, 'Chess kr-vs-kp', 2),
    'D12': (15, 'Breast Cancer Wisconsin Original', 2),
    'D13': (27, 'Credit Approval', 2),
    'D14': (43, 'Haberman Survival', 2),
    'D15': (46, 'Hepatitis', 2),
    'D16': (73, 'Mushroom', 2),
    'D17': (78, 'Page Blocks', 5),
    'D18': (80, 'Optical Digits', 10),
    'D19': (94, 'Spambase', 2),
    'D20': (144, 'German Credit', 2),
    'D21': (146, 'Landsat Satellite', 6),
    'D22': (159, 'MAGIC Gamma', 2),
    'D23': (161, 'Mammographic Mass', 2),
    'D24': (174, 'Parkinsons', 2),
    'D25': (176, 'Blood Transfusion', 2),
    'D26': (212, 'Vertebral Column', 3),
    'D27': (327, 'Phishing Websites', 2),
    'D28': (50, 'Image Segmentation', 7),
}


def generate_mixed_gumbel(n, loc1, scale1, loc2, scale2, mix_ratio=0.5):
    n1 = int(n * mix_ratio)
    x1 = gumbel_r.rvs(loc=loc1, scale=scale1, size=n1)
    n2 = n - n1
    x2 = gumbel_r.rvs(loc=loc2, scale=scale2, size=n2)
    x = np.concatenate([x1, x2])
    np.random.shuffle(x)
    return x


def beta_sd(alpha):
    return 0.5 / np.sqrt(2.0 * alpha + 1.0)


def shape_component(level, n, loc):
    # Beta(alpha, alpha) rescaled to the mean and SD of Beta(0.2, 0.2); level 'N' is the Gaussian limit
    if level == 'N':
        return np.random.normal(loc + 0.5, beta_sd(0.2), n)
    alpha = float(level)
    b = np.random.beta(alpha, alpha, n)
    if alpha == 0.2:
        return loc + b  # D10 itself, without rounding differences
    return loc + 0.5 + (b - 0.5) * (beta_sd(0.2) / beta_sd(alpha))


def synthetic_dataset(id):
    if id == 'D1':
        n1, n2, n3 = 150, 100, 50
        nClusters = 3

        x1 = np.random.normal(0, 0.22, n1)
        y1 = np.random.normal(0, 0.22, n1)

        x2 = np.random.normal(0.5, 0.16, n2)
        y2 = np.random.normal(0.5, 0.16, n2)

        x3 = np.random.normal(-0.25, 0.09, n3)
        y3 = np.random.normal(0.4, 0.09, n3)

        synthetic = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2)), np.column_stack((x3, y3))))
        data_ref = np.concatenate([np.full(n1, 0), np.full(n2, 1), np.full(n3, 2)])
        return [synthetic, data_ref, nClusters, 'Spherical Gaussians (2D)']

    elif id == 'D2':
        n1, n2, n3 = 100, 100, 100
        nClusters = 3

        x1 = np.random.normal(0, 0.18, n1)
        y1 = np.random.normal(0, 0.18, n1)
        z1 = np.random.normal(0, 0.18, n1)

        x2 = np.random.normal(0.2, 0.18, n2)
        y2 = np.random.normal(-0.2, 0.18, n2)
        z2 = np.random.normal(-0.2, 0.18, n2)

        x3 = np.random.normal(-0.2, 0.18, n3)
        y3 = np.random.normal(0.2, 0.18, n3)
        z3 = np.random.normal(0.2, 0.18, n3)

        synthetic = np.vstack((np.column_stack((x1, y1, z1)), np.column_stack((x2, y2, z2)),
                               np.column_stack((x3, y3, z3))))
        data_ref = np.concatenate([np.full(n1, 0), np.full(n2, 1), np.full(n3, 2)])
        return [synthetic, data_ref, nClusters, 'Spherical Gaussians (3D)']

    elif id == 'D3':
        n1, n2, n3 = 200, 100, 50
        nClusters = 3

        x1 = np.random.normal(0, 2.8, n1)
        y1 = np.random.normal(0, 0.05, n1)

        x2 = np.random.normal(0.3, 2.5, n2)
        y2 = np.random.normal(-0.13, 0.04, n2)

        x3 = np.random.normal(-0.3, 2.2, n3)
        y3 = np.random.normal(0.11, 0.04, n3)

        synthetic = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2)), np.column_stack((x3, y3))))
        data_ref = np.concatenate([np.full(n1, 0), np.full(n2, 1), np.full(n3, 2)])
        return [synthetic, data_ref, nClusters, 'Elliptical Gaussians (2D)']

    elif id == 'D4':
        n1, n2, n3 = 100, 100, 100
        nClusters = 3

        x1 = np.random.normal(0, 1.5, n1)
        y1 = np.random.normal(0, 0.04, n1)
        z1 = np.random.normal(0, 1.5, n1)

        x2 = np.random.normal(0.2, 1.5, n2)
        y2 = np.random.normal(-0.1, 0.04, n2)
        z2 = np.random.normal(0.2, 1.5, n2)

        x3 = np.random.normal(-0.2, 1.5, n3)
        y3 = np.random.normal(0.1, 0.04, n3)
        z3 = np.random.normal(-0.2, 1.5, n3)

        synthetic = np.vstack((np.column_stack((x1, y1, z1)), np.column_stack((x2, y2, z2)),
                               np.column_stack((x3, y3, z3))))
        data_ref = np.concatenate([np.full(n1, 0), np.full(n2, 1), np.full(n3, 2)])
        return [synthetic, data_ref, nClusters, 'Elliptical Gaussians (3D)']

    elif id == 'D5':
        n1, n2, n3 = 100, 100, 100
        nClusters = 3

        x1 = list(np.random.normal(0.2, 0.08, n1 // 2)) + list(np.random.normal(0.8, 0.1, n1 // 2))
        y1 = np.random.normal(0, 0.03, n1)

        x2 = list(np.random.normal(0.35, 0.08, n2 // 2)) + list(np.random.normal(0.95, 0.1, n2 // 2))
        y2 = np.random.normal(-0.15, 0.03, n2)

        x3 = np.random.normal(0.85, 0.05, n3)
        y3 = list(np.random.normal(-0.6, 0.1, n3 // 2)) + list(np.random.normal(-0.1, 0.1, n3 // 2))

        synthetic = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2)), np.column_stack((x3, y3))))
        data_ref = np.concatenate([np.full(n1, 0), np.full(n2, 1), np.full(n3, 2)])
        return [synthetic, data_ref, nClusters, 'Gaussian mixtures (2D)']

    elif id == 'D6':
        n1, n2, n3, n4 = 100, 100, 100, 100
        nClusters = 4

        x1 = np.random.normal(0.2, 0.02, n1)
        y1 = list(np.random.normal(-0.5, 0.12, n1 // 2)) + list(np.random.normal(0.1, 0.12, n1 // 2))
        z1 = list(np.random.normal(-0.5, 0.12, n1 // 2)) + list(np.random.normal(0.1, 0.12, n1 // 2))

        x2 = np.random.normal(0.35, 0.02, n2)
        y2 = list(np.random.normal(-0.5, 0.12, n2 // 2)) + list(np.random.normal(0.1, 0.12, n2 // 2))
        z2 = list(np.random.normal(-0.5, 0.12, n2 // 2)) + list(np.random.normal(0.1, 0.12, n2 // 2))

        x3 = list(np.random.normal(-0.3, 0.08, n3 // 2)) + list(np.random.normal(0.3, 0.1, n3 // 2))
        y3 = np.random.normal(0, 0.03, n3)
        z3 = np.random.normal(0, 0.03, n3)

        x4 = list(np.random.normal(-0.15, 0.08, n4 // 2)) + list(np.random.normal(0.45, 0.12, n4 // 2))
        y4 = np.random.normal(-0.1, 0.03, n4)
        z4 = np.random.normal(-0.1, 0.03, n4)

        synthetic = np.vstack((np.column_stack((x1, y1, z1)), np.column_stack((x2, y2, z2)),
                               np.column_stack((x3, y3, z3)), np.column_stack((x4, y4, z4))))
        data_ref = np.concatenate([np.full(n1, 0), np.full(n2, 1), np.full(n3, 2), np.full(n4, 3)])
        return [synthetic, data_ref, nClusters, 'Gaussian mixtures (3D)']

    elif id == 'D7':
        n1, n2, n3 = 100, 100, 100
        nClusters = 3

        x1 = np.random.beta(0.2, 0.2, n1)
        y1 = np.random.normal(0, 0.03, n1)

        x2 = 0.15 + np.random.beta(0.2, 0.2, n2)
        y2 = -0.15 + np.random.normal(0, 0.03, n2)

        x3 = 0.85 + np.random.normal(0, 0.05, n3)
        y3 = -0.5 + np.random.beta(2, 7, n3)

        synthetic = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2)), np.column_stack((x3, y3))))
        data_ref = np.concatenate([np.full(n1, 0), np.full(n2, 1), np.full(n3, 2)])
        return [synthetic, data_ref, nClusters, 'Beta + Normal']

    elif id == 'D8':
        n1, n2, n3 = 100, 100, 100
        nClusters = 3

        x1 = arcsine.rvs(size=n1)
        y1 = np.random.normal(0, 0.03, n1)

        x2 = 0.15 + arcsine.rvs(size=n2)
        y2 = -0.15 + np.random.normal(0, 0.03, n2)

        x3 = 0.85 + np.random.normal(0, 0.05, n3)
        y3 = -0.7 + arcsine.rvs(size=n3)

        synthetic = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2)), np.column_stack((x3, y3))))
        data_ref = np.concatenate([np.full(n1, 0), np.full(n2, 1), np.full(n3, 2)])
        return [synthetic, data_ref, nClusters, 'Arcsine + Normal']

    elif id == 'D9':
        n1, n2, n3 = 100, 100, 100
        nClusters = 3

        x1 = generate_mixed_gumbel(n1, 10.0, 1.8, 25.0, 2.2, mix_ratio=0.5)
        x2 = generate_mixed_gumbel(n2, 12.0, 2.0, 28.0, 1.5, mix_ratio=0.6)
        x3 = np.random.normal(25, 1, n1)

        y1 = np.random.normal(8, 1, n1)
        y2 = np.random.normal(0, 1, n2)
        y3 = generate_mixed_gumbel(n3, -8.0, 2.7, 10.0, 2.5, mix_ratio=0.4)

        synthetic = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2)), np.column_stack((x3, y3))))
        data_ref = np.concatenate([np.full(n1, 0), np.full(n2, 1), np.full(n3, 2)])
        return [synthetic, data_ref, nClusters, 'Gumbel mixtures + Normal']

    elif id == 'D10':
        n1, n2, n3 = 100, 100, 100
        nClusters = 3

        x1 = np.random.beta(0.2, 0.2, n1)
        y1 = np.random.normal(0, 0.03, n1)
        z1 = np.random.beta(0.2, 0.2, n1)

        x2 = 0.15 + np.random.beta(0.2, 0.2, n2)
        y2 = -0.15 + np.random.normal(0, 0.03, n2)
        z2 = 0.0 + np.random.beta(0.2, 0.2, n2)

        x3 = 0.85 + np.random.normal(0, 0.05, n3)
        y3 = -0.6 + np.random.beta(2, 7, n3)
        z3 = 0.5 + np.random.normal(0, 0.35, n3)

        synthetic = np.vstack((np.column_stack((x1, y1, z1)), np.column_stack((x2, y2, z2)),
                               np.column_stack((x3, y3, z3))))
        data_ref = np.concatenate([np.full(n1, 0), np.full(n2, 1), np.full(n3, 2)])
        return [synthetic, data_ref, nClusters, 'Beta + Normal (3D)']

    elif id in SHAPE:
        # same random calls, in the same order, as D10
        level = id.split('_')[1]
        n1, n2, n3 = 100, 100, 100
        nClusters = 3

        x1 = shape_component(level, n1, 0.0)
        y1 = np.random.normal(0, 0.03, n1)
        z1 = shape_component(level, n1, 0.0)

        x2 = shape_component(level, n2, 0.15)
        y2 = -0.15 + np.random.normal(0, 0.03, n2)
        z2 = shape_component(level, n2, 0.0)

        x3 = 0.85 + np.random.normal(0, 0.05, n3)
        y3 = -0.6 + np.random.beta(2, 7, n3)
        z3 = 0.5 + np.random.normal(0, 0.35, n3)

        synthetic = np.vstack((np.column_stack((x1, y1, z1)), np.column_stack((x2, y2, z2)),
                               np.column_stack((x3, y3, z3))))
        data_ref = np.concatenate([np.full(n1, 0), np.full(n2, 1), np.full(n3, 2)])
        return [synthetic, data_ref, nClusters, f'D10 with shape {level} (3D)']

    raise ValueError(f'Unknown synthetic dataset: {id}')


def download_uci(id):
    from ucimlrepo import fetch_ucirepo

    uci_id = REAL[id][0]
    ds = fetch_ucirepo(id=uci_id)
    feats = ds.data.features.copy()
    targets = ds.data.targets
    if isinstance(targets, pd.Series):
        targets = targets.to_frame()

    original = [str(c) for c in feats.columns]
    declared = set()
    for _, row in ds.variables.iterrows():
        if (str(row.get('name')) in original and str(row.get('role', '')).strip().lower() == 'feature'
                and str(row.get('type', '')).strip().lower() == 'categorical'):
            declared.add(str(row.get('name')))

    occurrences = {}
    unique = []
    for name in original:
        k = occurrences.get(name, 0)
        unique.append(name if k == 0 else f'{name}__dup{k}')
        occurrences[name] = k + 1
    feats.columns = unique

    cat_names = {u for o, u in zip(original, unique) if o in declared}
    for position, name in enumerate(feats.columns):
        dtype = feats.iloc[:, position].dtype
        if (pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)):
            cat_names.add(name)

    cont_cols = [c for c in feats.columns if c not in cat_names]
    cat_cols = [c for c in feats.columns if c in cat_names]

    if cont_cols:
        keep = feats[cont_cols].apply(pd.to_numeric, errors='coerce').notna().all(axis=1).to_numpy()
    else:
        keep = np.ones(len(feats), dtype=bool)
    feats = feats.loc[keep].reset_index(drop=True)
    targets = targets.loc[keep].reset_index(drop=True)

    out = pd.DataFrame(index=range(len(feats)))
    for c in cont_cols:
        out[c] = pd.to_numeric(feats[c], errors='coerce').astype(float)
    for c in cat_cols:
        out['cat_' + c] = feats[c].astype('string')
    if 'class' in out.columns:
        raise ValueError(f'{id}: a feature is named "class".')
    out['class'] = targets.iloc[:, 0].to_numpy()

    UCI_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(UCI_DIR / f'{id}.csv', index=False)


def read_uci(id):
    path = UCI_DIR / f'{id}.csv'
    if not path.exists():
        download_uci(id)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[''])
    cont_cols = [c for c in df.columns if c != 'class' and not c.startswith('cat_')]
    cat_cols = [c for c in df.columns if c.startswith('cat_')]
    cont = df[cont_cols].to_numpy(dtype=float) if cont_cols else np.empty((len(df), 0))
    cat_df = df[cat_cols].rename(columns=lambda c: c[4:]).astype('string')
    ref = pd.factorize(df['class'])[0].tolist()
    return cont, cat_df, ref, cont_cols


def remove_constant_columns(X):
    X = np.asarray(X, dtype=float)
    if X.shape[1] == 0:
        return X
    variances = np.var(X, axis=0)
    keep = np.isfinite(variances) & (variances > 0.0)
    return X[:, keep]


def scale_non_binary(X):
    is_binary = np.array([len(np.unique(X[:, j])) <= 2 for j in range(X.shape[1])])
    if is_binary.all():
        return X
    Xc = X.copy()
    Xc[:, ~is_binary] = StandardScaler().fit_transform(X[:, ~is_binary])
    return Xc


def one_hot(cat_df):
    if cat_df.shape[1] == 0:
        return np.empty((len(cat_df), 0))
    pieces = []
    for c in cat_df.columns:
        s = cat_df[c].astype('string')
        if s.nunique(dropna=True) <= 2:
            pieces.append(pd.factorize(s)[0].astype(float).reshape(-1, 1))
        else:
            d = pd.get_dummies(s, prefix=str(c), dummy_na=False).astype(float)
            pieces.append(d.to_numpy())
    return remove_constant_columns(np.hstack(pieces))


def preprocess(cont, cat_df):
    cont = remove_constant_columns(cont)
    cat_block = one_hot(cat_df)

    if cont.shape[1] > 0:
        cont = scale_non_binary(cont)
        if cont.shape[0] >= 2:
            cont = PCA(n_components=min(5, cont.shape[1]), whiten=False, random_state=SEED).fit_transform(cont)

    blocks = [b for b in (cont, cat_block) if b.shape[1] > 0]
    return np.hstack(blocks)


def real_dataset(id):
    cont, cat_df, ref, _ = read_uci(id)
    return [preprocess(cont, cat_df), ref, REAL[id][2], REAL[id][1]]
