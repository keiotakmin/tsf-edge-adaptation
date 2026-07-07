"""M5/W1 (referee): build ADDITIONAL BDG2 subsets to test (a) external validity beyond site
Rat, (b) whether the paper's "BDG2 is flat / adaptation barely pays" byproduct is an artifact
of selecting the 15 LEAST-missing buildings (the cleanest meters may be the most stable ones),
and (c) SCALE (referee W1: hundreds of meters, not 15).

From the raw BDG2 hourly electricity table (data/bdg2_electricity_cleaned.csv, Zenodo
10.5281/zenodo.3887306), using the same recipe as the shipped bdg2.csv (2 years hourly,
ex-ante ffill->bfill, 'date' column dropped by load_csv):
  bdg2_fox.csv        site Fox      15 least-missing buildings  (2nd site, selection as before)
  bdg2_panther.csv    site Panther  15 least-missing buildings  (3rd site)
  bdg2_rat_worst.csv  site Rat      the 15 MOST-missing buildings with >=50% coverage
                                    (ANTI-selection: directly tests the selection-bias concern)
  bdg2_rat_all.csv    site Rat      ALL meters with >=50% coverage (one full site)
  bdg2_fleet.csv      ALL sites     up to 15 least-missing meters per site (fleet-scale,
                                    ~hundreds of channels across the whole corpus)
Sanity: reproduces the shipped bdg2.csv building set from the same rule (least-missing 15 at
Rat) and reports the overlap.
"""
from __future__ import annotations
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "bdg2_electricity_cleaned.csv")
OUTDIR = os.path.join(ROOT, "experiments", "tsf_edge", "data")

df = pd.read_csv(RAW)
ts = df.columns[0]
df[ts] = pd.to_datetime(df[ts])
print(f"raw: {df.shape[0]} rows x {df.shape[1]-1} meters, {df[ts].min()} .. {df[ts].max()}")

site_of = {c: c.split("_")[0] for c in df.columns[1:]}
miss = df.iloc[:, 1:].isna().mean()


def build(site, n=15, worst=False, min_cov=0.5):
    cols = [c for c in df.columns[1:] if site_of[c] == site]
    m = miss[cols]
    if worst:
        eligible = m[m <= (1 - min_cov)]                  # keep meters with >= min_cov coverage
        chosen = eligible.sort_values(ascending=False).head(n).index.tolist()
    else:
        chosen = m.sort_values(ascending=True).head(n).index.tolist()
    sub = df[[ts] + chosen].rename(columns={ts: "date"})
    sub[chosen] = sub[chosen].ffill().bfill()             # ex-ante fill, fixed before modeling
    tag = "worst" if worst else "least"
    print(f"{site:8s} ({tag:5s}-missing {n}): missingness "
          f"{m[chosen].min():.3%} .. {m[chosen].max():.3%}  -> {len(sub)} rows")
    return sub, chosen


# sanity: the shipped bdg2.csv should equal the least-missing-15 rule at Rat
shipped = pd.read_csv(os.path.join(OUTDIR, "bdg2.csv"), nrows=0).columns[1:].tolist()
_, rat_best = build("Rat")
overlap = len(set(shipped) & set(rat_best))
print(f"sanity: shipped bdg2.csv vs least-missing-15 rule at Rat: {overlap}/15 overlap")

for site, worst, fname in [("Fox", False, "bdg2_fox.csv"),
                           ("Panther", False, "bdg2_panther.csv"),
                           ("Rat", True, "bdg2_rat_worst.csv")]:
    sub, chosen = build(site, worst=worst)
    sub.to_csv(os.path.join(OUTDIR, fname), index=False)
    print(f"  wrote {fname}: {chosen[:3]} ...")


DEAD_STD = 1e-2   # ex-ante inactive-meter rule: a meter whose readings are essentially
                  # constant over the first half (the commissioning/warmup pool) cannot be
                  # z-normalized or forecast; one such meter (std 2e-6) exploded under
                  # normalization and silently flattened the whole fleet-level metric.


def build_multi(cols, fname):
    sub = df[[ts] + cols].rename(columns={ts: "date"})
    sub[cols] = sub[cols].ffill().bfill()
    half = len(sub) // 2
    std = sub[cols].iloc[:half].std()
    dead = std[std < DEAD_STD].index.tolist()
    if dead:
        print(f"  {fname}: excluding {len(dead)} inactive meter(s) "
              f"(first-half std < {DEAD_STD:g}): {dead}")
        cols = [c for c in cols if c not in dead]
        sub = sub[["date"] + cols]
    sub.to_csv(os.path.join(OUTDIR, fname), index=False)
    print(f"  wrote {fname}: {len(cols)} meters, missingness "
          f"{miss[cols].min():.2%} .. {miss[cols].max():.2%}")


# W1 scale subsets ------------------------------------------------------------
MIN_COV = 0.5
sites = sorted({s for s in site_of.values()})
rat_all = [c for c in df.columns[1:] if site_of[c] == "Rat" and miss[c] <= 1 - MIN_COV]
build_multi(rat_all, "bdg2_rat_all.csv")

fleet = []
for s in sites:
    cols = [c for c in df.columns[1:] if site_of[c] == s and miss[c] <= 1 - MIN_COV]
    fleet += miss[cols].sort_values().head(15).index.tolist()
print(f"fleet: {len(sites)} sites -> {len(fleet)} meters (up to 15 least-missing per site, "
      f">= {MIN_COV:.0%} coverage)")
build_multi(fleet, "bdg2_fleet.csv")
