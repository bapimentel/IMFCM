import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import time
import urllib.request
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score

from algorithms import (FCM, FCM_D, FCM_MFCM, FCM_S, FCM_WD, GK, IMFCM, M_GRID, MFCM,
                        compute_interpretation_indices, recommend_m)
from analysis import FR_Index, SCATTER_COLORS, SCATTER_MARKERS, SCATTER_PANEL, SCATTER_STYLE


APP_DIR = Path(__file__).resolve().parent
DATA = APP_DIR / "data" / "brca" / "brca_tcga.csv"
RESULTS = APP_DIR / "results" / "application"
TAB_DIR = RESULTS / "tables"
FIG_DIR = RESULTS / "figures"

API = "https://www.cbioportal.org/api"
STUDY = "brca_tcga_pan_can_atlas_2018"
LEGACY = "brca_tcga"
GENES = {2099: "ESR1", 5241: "PGR", 2064: "ERBB2", 2886: "GRB7", 4288: "MKI67"}
FEATURES = list(GENES.values())
IHC = ["ER_STATUS_BY_IHC", "PR_STATUS_BY_IHC", "IHC_HER2", "HER2_FISH_STATUS"]
SUBTYPES = ["LumA", "LumB", "Her2", "Basal"]
K = 4
SEED, N_REP, N_REP_RESAMPLE = 42, 100, 20
N_BOOT_M, N_BOOT = 30, 200

EXPECTED = pd.DataFrame({
    "ESR1": [+1, +1, -1, -1], "PGR": [+1, 0, -1, -1], "ERBB2": [0, 0, +1, -1], "GRB7": [0, 0, +1, 0],
    "MKI67": [-1, +1, +1, +1],
}, index=SUBTYPES)

COMPETITORS = {"FCM": FCM, "GK": GK, "MFCM": MFCM, "FCM-D": FCM_D, "FCM-S": FCM_S, "FCM-WD": FCM_WD,
               "FCM-MFCM": lambda X, c, m: FCM_MFCM(X, c, m, 0.5)}


def fetch(url, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json",
                                                              "Accept": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def pivot_clinical(records):
    c = pd.DataFrame(records)
    return c.pivot(index="patientId", columns="clinicalAttributeId", values="value")


def build_table(expression, clinical, legacy):
    e = pd.DataFrame(expression)
    e = e[e["sampleId"].str.endswith("-01")]
    X = e.pivot(index="patientId", columns="entrezGeneId", values="value").rename(columns=GENES)[FEATURES]
    clin = pivot_clinical(clinical)[["SUBTYPE"]]
    ihc = pivot_clinical(legacy)[IHC]
    return X.join(clin, how="inner").join(ihc, how="left").dropna(subset=FEATURES + ["SUBTYPE"])


def download():
    expression = fetch(f"{API}/molecular-profiles/{STUDY}_rna_seq_v2_mrna/molecular-data/fetch?projection=SUMMARY",
                       {"entrezGeneIds": list(GENES), "sampleListId": f"{STUDY}_all"})
    clinical = fetch(f"{API}/studies/{STUDY}/clinical-data?clinicalDataType=PATIENT&projection=SUMMARY")
    legacy = fetch(f"{API}/studies/{LEGACY}/clinical-data?clinicalDataType=PATIENT&projection=SUMMARY")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    build_table(expression, clinical, legacy).to_csv(DATA, index_label="patientId")


def load():
    if not DATA.exists():
        download()
    df = pd.read_csv(DATA, index_col="patientId", keep_default_na=False, na_values=[""],
                     float_precision="round_trip")
    df[FEATURES] = np.log2(df[FEATURES] + 1)
    df["subtype"] = df["SUBTYPE"].str.replace("BRCA_", "", regex=False)
    df = df[df["subtype"].isin(SUBTYPES)].copy()
    pos = lambda s: s.map({"Positive": True, "Negative": False})
    df["ER_ihc"], df["PR_ihc"] = pos(df["ER_STATUS_BY_IHC"]), pos(df["PR_STATUS_BY_IHC"])
    her2 = pos(df["IHC_HER2"])
    fish = pos(df["HER2_FISH_STATUS"])
    df["HER2_clin"] = her2.where(df["IHC_HER2"] != "Equivocal", fish)
    hr = df["ER_ihc"] | df["PR_ihc"]
    ok = df[["ER_ihc", "PR_ihc", "HER2_clin"]].notna().all(axis=1)
    df["ihc_group"] = np.where(~ok, None, np.where(df["HER2_clin"] == True, "HER2+",
                                                  np.where(hr == True, "HR+/HER2-", "Triple negative")))
    return df.reset_index()


def standardize(df, features=FEATURES):
    return ((df[features] - df[features].mean()) / df[features].std()).to_numpy(float)


def fit_imfcm(X, m, k=K, n_rep=N_REP, seed=SEED):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_rep):
        J, labels, memb, count, U, P, A = IMFCM(X, list(rng.choice(len(X), size=k, replace=False)), m)
        if best is None or J < best["J"]:
            best = dict(J=float(J), U_list=U, memb=memb, labels=labels, P=P, A=A, m=m)
    return best


def fit_competitor(name, X, m=2.0, k=K, n_rep=N_REP, seed=SEED):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    best = None
    for _ in range(n_rep):
        c = [int(i) for i in rng.choice(len(X), size=k, replace=False)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = COMPETITORS[name](X, c, m)
        except Exception:
            continue
        if np.isfinite(r[0]) and (best is None or r[0] < best[0]):
            best = r
    if best is None:
        return None
    memb = np.asarray(best[2], float)
    return dict(labels=np.argmax(memb, axis=1), memb=memb)


def bootstrap_J(args):
    X, idx, m, seed = args
    return fit_imfcm(X[idx], m, K, N_REP_RESAMPLE, seed)["J"]


def select_m(X, pool):
    rng = np.random.default_rng(SEED)
    boots = [rng.choice(len(X), size=len(X), replace=True) for _ in range(N_BOOT_M)]
    J = np.array(pool.map(bootstrap_J, [(X, i, m, SEED + b) for m in M_GRID for b, i in enumerate(boots)]))
    sel = recommend_m(M_GRID, J.reshape(len(M_GRID), N_BOOT_M).mean(axis=1))
    sel["J"] = J.reshape(len(M_GRID), N_BOOT_M)
    return sel


def bootstrap_one(args):
    X, ref, m, b = args
    rng = np.random.default_rng(SEED + 1000 + b)
    idx = np.unique(rng.choice(len(X), size=len(X), replace=True))
    r = fit_imfcm(X[idx], m, K, N_REP_RESAMPLE, SEED + b)
    M = np.array([[np.sum((ref[idx] == a) & (r["labels"] == c)) for c in range(K)] for a in range(K)])
    _, perm = linear_sum_assignment(-M)
    ind = compute_interpretation_indices(r["U_list"], X[idx], r["P"], m)
    lb, r_ = r["labels"], ref[idx]
    jac = [max(((r_ == c) & (lb == j)).sum() / ((r_ == c) | (lb == j)).sum() for j in range(K)) for c in range(K)]
    return dict(A=r["A"][perm], CTR=ind["CTR_ij"][perm], ari=adjusted_rand_score(r_, lb), jac=jac)


def fit_m(args):
    X, m = args
    return m, fit_imfcm(X, m, K, 50, SEED)


def match_subtypes(labels, y, k=K):
    C = np.array([[np.sum((labels == c) & (y == s)) for s in SUBTYPES] for c in range(k)])
    r, c = linear_sum_assignment(-C)
    return {int(a): SUBTYPES[b] for a, b in zip(r, c)}


def ordered(mat, mapping):
    inv = {v: k for k, v in mapping.items()}
    return np.vstack([mat[inv[s]] for s in SUBTYPES])


def agreement(labels, y, memb=None):
    mp = match_subtypes(labels, y)
    pred = np.array([mp[int(l)] for l in labels])
    out = {"ARI": adjusted_rand_score(y, labels), "NMI": normalized_mutual_info_score(y, labels),
           "Accuracy (matched)": float(np.mean(pred == y))}
    if memb is not None:
        out["FR"] = float(FR_Index(memb, pd.factorize(y)[0]))
    return out, mp, pred


def class_statistics(Xdf, y):
    d, homog = {}, {}
    for s in SUBTYPES:
        a, b = Xdf[y == s], Xdf[y != s]
        pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
        d[s] = ((a.mean() - b.mean()) / pooled).abs()
        homog[s] = 1 / a.std()
    return pd.DataFrame(d).T, pd.DataFrame(homog).T


def rank_agreement(A, B):
    rows = {i: spearmanr(A.loc[i], B.loc[i]).statistic for i in A.index}
    rows["All cells"] = spearmanr(A.to_numpy().ravel(), B.to_numpy().ravel()).statistic
    return pd.Series(rows)


def signature_agreement(profile):
    mask = EXPECTED.to_numpy() != 0
    return float((np.sign(profile.loc[SUBTYPES].to_numpy())[mask] == EXPECTED.to_numpy()[mask]).mean())


def save(df, name, index=True):
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB_DIR / f"{name}.csv", index=index)


def save_figure(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_m_selection(sel, m_used):
    with plt.rc_context(SCATTER_STYLE):
        fig, ax1 = plt.subplots(figsize=(2 * SCATTER_PANEL + 1.0, SCATTER_PANEL + 0.3))
        ax1.plot(sel["grid"], sel["Jbar"], marker="o", color=SCATTER_COLORS[0], label=r"$\bar{J}$")
        ax1.set_xlabel("Fuzzifier $m$")
        ax1.set_ylabel(r"Mean objective function $\bar{J}$")
        ax2 = ax1.twinx()
        for key, lab, mk, col in (("d1", r"$d\bar{J}/dm$", "x", SCATTER_COLORS[1]),
                                  ("d2", r"$d^2\bar{J}/dm^2$", "s", SCATTER_COLORS[2]),
                                  ("d3", r"$d^3\bar{J}/dm^3$", "^", SCATTER_COLORS[3])):
            ax2.plot(sel["grid"], sel[key], marker=mk, color=col, linestyle="--", linewidth=0.9, label=lab)
        ax2.set_ylabel("Derivatives")
        ax2.grid(False)
        ax1.axvline(m_used, color="0.3", linestyle=":", linewidth=1)
        ax1.text(m_used, ax1.get_ylim()[1], rf" $\hat m^{{(3)}}={m_used:g}$", va="top", fontsize=8)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper right")
        fig.tight_layout()
        save_figure(fig, "brca_m_selection")


def heat(ax, M, rows, cols, fmt, cmap, center):
    M = np.asarray(M, float)
    lim = np.nanmax(np.abs(M))
    ax.imshow(M, cmap=cmap, aspect="auto", vmin=-lim if center else None, vmax=lim if center else None)
    ax.set_xticks(range(len(cols)), cols, rotation=0)
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7)


def plot_heatmaps(panels, name, fmt="{:.2f}", cmap="RdBu_r", center=True):
    with plt.rc_context(SCATTER_STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(len(panels) * (SCATTER_PANEL + 0.5) + 0.6, 2.2),
                                 squeeze=False)
        for ax, (t, df) in zip(axes[0], panels):
            heat(ax, df.to_numpy(), list(df.index), list(df.columns), fmt, cmap, center)
            ax.set_title(t, fontsize=9, pad=6)
        fig.tight_layout()
        save_figure(fig, name)


def plot_coherence(ctr, d, a, homog, rho_ctr, rho_a):
    with plt.rc_context(SCATTER_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(2 * SCATTER_PANEL + 2.0, SCATTER_PANEL + 0.9))
        for k, s in enumerate(SUBTYPES):
            axes[0].scatter(d.loc[s], ctr.loc[s], marker=SCATTER_MARKERS[k], c=SCATTER_COLORS[k], s=30,
                            edgecolors="black", linewidths=0.3, label=s)
            axes[1].scatter(homog.loc[s], a.loc[s], marker=SCATTER_MARKERS[k], c=SCATTER_COLORS[k], s=30,
                            edgecolors="black", linewidths=0.3, label=s)
            for v in ctr.columns:
                axes[0].annotate(v, (d.loc[s, v], ctr.loc[s, v]), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
                axes[1].annotate(v, (homog.loc[s, v], a.loc[s, v]), fontsize=5.5, xytext=(2, 2),
                                 textcoords="offset points")
        axes[0].set_xlabel("|d| (subtype vs rest)")
        axes[0].set_ylabel(r"$\mathrm{CTR}(i,j)$")
        axes[0].set_title(f"(a) Spearman = {rho_ctr:.2f}", fontsize=9, y=-0.4)
        axes[1].set_xlabel("1/SD within subtype")
        axes[1].set_ylabel(r"$a_{ij}$")
        axes[1].set_title(f"(b) Spearman = {rho_a:.2f}", fontsize=9, y=-0.4)
        axes[0].legend(fontsize=6.5, frameon=False, loc="best")
        fig.tight_layout()
        save_figure(fig, "brca_interpretation_coherence")


def plot_memberships(Xdf, y, memb, correct):
    with plt.rc_context(SCATTER_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(2 * SCATTER_PANEL + 0.8, SCATTER_PANEL + 0.35))
        top = memb.max(axis=1)
        for k, s in enumerate(SUBTYPES):
            m = y == s
            axes[0].scatter(Xdf.loc[m, "ESR1"], Xdf.loc[m, "ERBB2"], marker=SCATTER_MARKERS[k], c=SCATTER_COLORS[k],
                            s=12, alpha=np.clip(top[m], 0.2, 1), edgecolors="black", linewidths=0.2)
        axes[0].set_xlabel("ESR1 (z)")
        axes[0].set_ylabel("ERBB2 (z)")
        axes[0].set_title("(a) Opacity = largest membership", fontsize=9, y=-0.52)
        data = [top[correct], top[~correct]]
        axes[1].boxplot(data, widths=0.5, showfliers=False)
        rng = np.random.default_rng(0)
        for j, v in enumerate(data, start=1):
            axes[1].scatter(np.full(len(v), j) + rng.uniform(-0.12, 0.12, len(v)), v, s=6,
                            c=SCATTER_COLORS[0 if j == 1 else 3], alpha=0.5, edgecolors="none")
        axes[1].set_xticks([1, 2], [f"Concordant\n(n={correct.sum()})", f"Discordant\n(n={(~correct).sum()})"])
        axes[1].set_ylabel("Largest membership")
        axes[1].set_title("(b)", fontsize=9, y=-0.52)
        h = [Line2D([], [], linestyle="", marker=SCATTER_MARKERS[k], markerfacecolor=SCATTER_COLORS[k],
                    markeredgecolor="black", markeredgewidth=0.3, markersize=5) for k in range(len(SUBTYPES))]
        axes[0].legend(h, SUBTYPES, fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                       ncol=4, handletextpad=0.2, columnspacing=0.8)
        fig.tight_layout()
        save_figure(fig, "brca_memberships")


def main():
    df = load()
    y = df["subtype"].to_numpy()
    X = standardize(df)
    Xdf = pd.DataFrame(X, columns=FEATURES)
    desc = df.groupby("subtype")[FEATURES].median().reindex(SUBTYPES)
    desc.insert(0, "n", df["subtype"].value_counts().reindex(SUBTYPES))
    save(desc, "data_medians_by_subtype")
    save(pd.crosstab(df["subtype"], df["ihc_group"]).reindex(SUBTYPES), "subtype_vs_ihc")

    pool = Pool(os.cpu_count())

    sel = select_m(X, pool)
    m = sel["m3"]
    save(pd.DataFrame({"m": sel["grid"], "mean J": sel["Jbar"], "d1": sel["d1"], "d2": sel["d2"], "d3": sel["d3"]}),
         "m_selection", index=False)
    plot_m_selection(sel, m)
    print(f"Recommended m: d1={sel['m1']} d2={sel['m2']} d3={m}", flush=True)

    fits = {"IMFCM": fit_imfcm(X, m, K, 100, SEED),
            **{n: fit_competitor(n, X, 2.0, K, 100, SEED) for n in COMPETITORS}}
    rows, maps = [], {}
    for name, f in fits.items():
        if f is None:
            continue
        a, mp, pred = agreement(f["labels"], y, f.get("memb"))
        maps[name] = (mp, pred)
        ihc = df["ihc_group"].notna().to_numpy()
        a["ARI (IHC groups)"] = adjusted_rand_score(df.loc[ihc, "ihc_group"], f["labels"][ihc])
        rows.append({"Method": name, **a})
    agree = pd.DataFrame(rows).set_index("Method")
    save(agree, "agreement")

    f, (mp, pred) = fits["IMFCM"], maps["IMFCM"]
    ind = compute_interpretation_indices(f["U_list"], X, f["P"], m)
    A = pd.DataFrame(ordered(f["A"], mp), index=SUBTYPES, columns=FEATURES)
    CTR = pd.DataFrame(ordered(ind["CTR_ij"], mp), index=SUBTYPES, columns=FEATURES)
    COR = pd.DataFrame(ordered(ind["COR_ij"], mp), index=SUBTYPES, columns=FEATURES)
    prof = pd.DataFrame(ordered(np.vstack([X[f["labels"] == c].mean(axis=0) for c in range(K)]), mp),
                        index=SUBTYPES, columns=FEATURES)
    clus = pd.DataFrame({"T_rel": ordered(ind["T_rel"][:, None], mp)[:, 0],
                         "J_rel": ordered(ind["J_rel"][:, None], mp)[:, 0],
                         "B_rel": ordered(ind["B_rel"][:, None], mp)[:, 0]}, index=SUBTYPES)
    var = pd.DataFrame({"COR_j": ind["COR_j"], "CTR_j": ind["CTR_j"]}, index=FEATURES)
    for nm, t in (("A", A), ("CTR_ij", CTR), ("COR_ij", COR), ("profiles_z", prof), ("clusters", clus),
                  ("variables", var)):
        save(t.round(4), f"interpretation_{nm}")
    true_prof = Xdf.groupby(y).mean().reindex(SUBTYPES)
    plot_heatmaps([("(a) PAM50 subtype", true_prof), ("(b) IMFCM clusters", prof)], "brca_profiles")
    plot_heatmaps([(r"(a) Feature weights $a_{ij}$", A), (r"(b) $\mathrm{CTR}(i,j)$", CTR)], "brca_imfcm_weights",
                  fmt="{:.3f}", cmap="Blues", center=False)

    cluster_names = np.array([mp[int(l)] for l in f["labels"]])
    ct = pd.crosstab(pd.Series(cluster_names, name="IMFCM cluster"),
                     pd.Series(df["ihc_group"].fillna("Not available").to_numpy(), name="IHC group")).reindex(SUBTYPES)
    save(ct, "clusters_vs_ihc")

    d, homog = class_statistics(Xdf, y)
    rho_ctr, rho_a = rank_agreement(CTR, d), rank_agreement(A, homog)
    save(pd.DataFrame({"Spearman(CTR, |d|)": rho_ctr, "Spearman(a_ij, 1/SD)": rho_a}), "rank_agreement")
    save(d.round(3), "true_cohen_d")
    save(homog.round(3), "true_homogeneity")
    plot_coherence(CTR, d, A, homog, rho_ctr["All cells"], rho_a["All cells"])

    boot = pool.map(bootstrap_one, [(X, f["labels"], m, b) for b in range(N_BOOT)])
    inv = {v: k for k, v in mp.items()}
    order = [inv[s] for s in SUBTYPES]
    Ab = np.array([b["A"] for b in boot])[:, order]
    CTRb = np.array([b["CTR"] for b in boot])[:, order]
    ci = lambda arr: pd.DataFrame([[f"{np.median(arr[:, i, j]):.3f} ({np.percentile(arr[:, i, j], 2.5):.3f}-"
                                    f"{np.percentile(arr[:, i, j], 97.5):.3f})" for j in range(len(FEATURES))]
                                   for i in range(K)], index=SUBTYPES, columns=FEATURES)
    save(ci(Ab), "bootstrap_A_ci")
    save(ci(CTRb), "bootstrap_CTR_ci")
    jac = np.mean([b["jac"] for b in boot], axis=0)[order]
    top = pd.DataFrame({
        "Top a_ij": [FEATURES[int(np.argmax(A.iloc[i]))] for i in range(K)],
        "P(top a_ij kept)": [float(np.mean(Ab[:, i].argmax(1) == np.argmax(A.iloc[i]))) for i in range(K)],
        "Lowest a_ij": [FEATURES[int(np.argmin(A.iloc[i]))] for i in range(K)],
        "P(lowest a_ij kept)": [float(np.mean(Ab[:, i].argmin(1) == np.argmin(A.iloc[i]))) for i in range(K)],
        "Top CTR": [FEATURES[int(np.argmax(CTR.iloc[i]))] for i in range(K)],
        "P(top CTR kept)": [float(np.mean(CTRb[:, i].argmax(1) == np.argmax(CTR.iloc[i]))) for i in range(K)],
        "Jaccard": jac}, index=SUBTYPES)
    save(top, "bootstrap_top_variable")

    rows = []
    neighbours = [g for g in sel["grid"] if abs(g - m) <= 0.55]
    for mm, r in pool.map(fit_m, [(X, g) for g in neighbours]):
        a, mp2, _ = agreement(r["labels"], y, r["memb"])
        i2 = compute_interpretation_indices(r["U_list"], X, r["P"], mm)
        A2 = ordered(r["A"], mp2)
        CTR2 = ordered(i2["CTR_ij"], mp2)
        rows.append({"m": mm, **a, "median largest membership": float(np.median(r["memb"].max(1))),
                     "max a_ij": float(r["A"].max()),
                     "Spearman(a_ij, main)": spearmanr(A2.ravel(), A.to_numpy().ravel()).statistic,
                     "Spearman(CTR, main)": spearmanr(CTR2.ravel(), CTR.to_numpy().ravel()).statistic,
                     "Spearman(a_ij, 1/SD)": spearmanr(A2.ravel(), homog.to_numpy().ravel()).statistic})
    across = pd.DataFrame(rows).sort_values("m")
    save(across, "interpretation_across_m", index=False)
    pool.close()
    pool.join()

    top_m = f["memb"].max(axis=1)
    correct = pred == y
    mw = mannwhitneyu(top_m[correct], top_m[~correct], alternative="greater")
    auc = roc_auc_score(~correct, -top_m)
    memb_tab = pd.DataFrame({"median largest membership (concordant)": [float(np.median(top_m[correct]))],
                             "median largest membership (discordant)": [float(np.median(top_m[~correct]))],
                             "n discordant": [int((~correct).sum())], "Mann-Whitney p": [mw.pvalue], "AUC": [auc]})
    save(memb_tab, "membership_vs_subtype", index=False)
    plot_memberships(Xdf, y, f["memb"], correct)

    luminal = {("LumA", "LumB"), ("LumB", "LumA")}
    basal = (cluster_names == "Basal") & df["ER_ihc"].notna().to_numpy()
    er = df["ER_ihc"].to_numpy()
    summary = {"n": len(df), "n complete IHC": int(df["ihc_group"].notna().sum()),
               "m": {"d1": sel["m1"], "d2": sel["m2"], "d3": m},
               "agreement": agree.round(4).to_dict(orient="index"),
               "signature_agreement": {"IMFCM": signature_agreement(prof), "PAM50": signature_agreement(true_prof)},
               "rank_agreement": {"CTR_vs_d": rho_ctr.round(3).to_dict(), "A_vs_homog": rho_a.round(3).to_dict()},
               "a_ij range": [float(A.to_numpy().min()), float(A.to_numpy().max())],
               "top-3 CTR share": {s: float(np.sort(CTR.loc[s].to_numpy())[-3:].sum()) for s in SUBTYPES},
               "luminal A/B swaps among discordant": int(sum((s, c) in luminal for s, c in zip(y, cluster_names))),
               "Basal cluster membership (median), ER+ vs ER- by IHC": [
                   float(np.median(top_m[basal & (er == True)])), float(np.median(top_m[basal & (er == False)]))],
               "median_boot_ARI": float(np.median([b["ari"] for b in boot])), "membership": memb_tab.iloc[0].to_dict()}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Application finished in {time.time() - start:.0f} s", flush=True)
