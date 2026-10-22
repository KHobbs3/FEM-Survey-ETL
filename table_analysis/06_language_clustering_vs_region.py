"""
06_language_clustering_vs_region.py
====================================
Unsupervised check on an optional region grouping (PROVINCE_TO_REGION in src/config.py).
Only meaningful where several survey languages map onto regions (Benin). Skipped when
PROVINCE_TO_REGION is None (Togo: one language, results reported by department instead).

Question: if we cluster respondents purely on which language(s) they prefer for
radio (`radio_language`, the only per-respondent language field in the survey),
do the resulting clusters line up with the region groups above -- i.e. is
"dominant local language" already implicit in the region grouping, or does it
cut across it?

Method: one-hot encode each respondent's selected radio language(s), run
k-means with k = number of assumed region groups plus a silhouette scan
over k=2..6 for context, then cross-tabulate cluster assignment against the
assumed region and report agreement (Adjusted Rand Index / Normalized Mutual
Information) plus a per-cluster language & region profile.

No individual-level data is exported -- only cluster-level aggregates -- consistent
with the rest of the pipeline.

Output: data/3_final/language_region_clustering_26_08_24.csv
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

from src.config import DIR_WEIGHTED_DATA, DIR_FINAL, PROVINCE_TO_REGION

N_CLUSTERS = len(set((PROVINCE_TO_REGION or {}).values()))  # one per assumed region group, for direct comparability
RANDOM_STATE = 42



def _clean_language(text):
    """'Ewe/ Eʋegbe' -> 'Ewe' (French half, same convention as _strip_hausa /
    _clean_bilingual elsewhere in the pipeline)."""
    text = str(text).strip()
    if "/" in text:
        fr = text.split("/", 1)[0].strip()
        return fr if fr else text
    return text


def main():
    if not PROVINCE_TO_REGION:
        print("PROVINCE_TO_REGION is not set in src/config.py -- no region grouping to test; skipping.")
        return
    data = pd.read_csv(DIR_WEIGHTED_DATA, low_memory=False)
    data["region"] = data["province"].map(PROVINCE_TO_REGION)

    # ── Build respondent x language binary matrix ──────────────────────────────
    lang_sets = (
        data["radio_language"].dropna().astype(str)
        .str.split("|")
        .apply(lambda vals: sorted({_clean_language(v) for v in vals if _clean_language(v)}))
    )
    valid_idx = lang_sets.index
    all_languages = sorted({lang for langs in lang_sets for lang in langs})

    X = pd.DataFrame(0, index=valid_idx, columns=all_languages, dtype=int)
    for idx, langs in lang_sets.items():
        X.loc[idx, langs] = 1

    weights = data.loc[valid_idx, "normalized_weight"]
    regions = data.loc[valid_idx, "region"]
    provinces = data.loc[valid_idx, "province"]

    print(f"Respondents with >=1 radio language selected: {len(X)} / {len(data)}")
    print(f"Languages seen: {all_languages}")

    # ── Silhouette scan k=2..6, for context (not necessarily = N_CLUSTERS) ──────
    print("\nSilhouette scan (k=2..6):")
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        sil = silhouette_score(X, km.labels_)
        print(f"  k={k}: silhouette={sil:.3f}")

    # ── Final clustering at k=4, matching the 4 assumed region groups ───────────
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE).fit(X)
    cluster = pd.Series(km.labels_, index=X.index, name="cluster")

    # ── Agreement between language-only clusters and the assumed region label ───
    mask = regions.notna()
    ari = adjusted_rand_score(regions[mask], cluster[mask])
    nmi = normalized_mutual_info_score(regions[mask], cluster[mask])
    print(f"\nAgreement between language clusters (k={N_CLUSTERS}) and assumed region groups:")
    print(f"  Adjusted Rand Index (ARI):        {ari:.3f}  (0 = no better than chance, 1 = perfect match)")
    print(f"  Normalized Mutual Info (NMI):     {nmi:.3f}  (0 = independent, 1 = perfect match)")
    print("  (Low scores mean the region groups are NOT simply a proxy for dominant")
    print("   radio language -- the two are cutting across each other, not the same split.)")

    # ── Per-cluster profile: dominant languages + region/province composition ──
    rows = []
    for c in range(N_CLUSTERS):
        in_c = cluster == c
        n = int(in_c.sum())
        wn = float(weights[in_c].sum())

        lang_prev = (X[in_c].mul(weights[in_c], axis=0).sum() / weights[in_c].sum() * 100).sort_values(ascending=False)
        top_langs = "; ".join(f"{lang}: {pct:.1f}%" for lang, pct in lang_prev.head(3).items())

        region_prev = (
            regions[in_c].value_counts(normalize=True).mul(100).round(1)
        )
        region_str = "; ".join(f"{r}: {p:.1f}%" for r, p in region_prev.items())
        majority_region = region_prev.idxmax() if not region_prev.empty else ""
        purity = region_prev.max() if not region_prev.empty else np.nan

        province_prev = provinces[in_c].value_counts(normalize=True).mul(100).round(1)
        province_str = "; ".join(f"{p}: {v:.1f}%" for p, v in province_prev.items())

        rows.append({
            "cluster": c,
            "n": n,
            "weighted_n": round(wn, 1),
            "top_languages_weighted_pct": top_langs,
            "region_composition_pct": region_str,
            "majority_assumed_region": majority_region,
            "purity_vs_assumed_region_pct": purity,
            "province_composition_pct": province_str,
        })

    profile = pd.DataFrame(rows)
    profile.attrs["ari"] = ari
    profile.attrs["nmi"] = nmi

    out_path = DIR_FINAL / "language_region_clustering_26_08_24.csv"
    profile.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(profile.to_string(index=False))

    return profile


if __name__ == "__main__":
    main()
