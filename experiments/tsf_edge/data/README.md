# Datasets

Six public multivariate series. `get_data.sh` downloads the five that are not shipped and
verifies them against `checksums.sha256` — the checksums of the exact files used in the paper.

| File | Source | License | Shipped? |
|---|---|---|---|
| ETTh1/ETTh2/ETTm1/ETTm2.csv | [ETDataset](https://github.com/zhouhaoyi/ETDataset) (ETT-small) | see source repo | no (downloaded) |
| appliances.csv | [UCI Appliances energy prediction](https://archive.ics.uci.edu/dataset/374) (`energydata_complete.csv`, renamed) | CC BY 4.0 | no (downloaded) |
| bdg2.csv | derived from [Building Data Genome 2](https://github.com/buds-lab/building-data-genome-project-2) (Zenodo DOI [10.5281/zenodo.3887306](https://doi.org/10.5281/zenodo.3887306)) | MIT (source) | **yes** |
| bdg2_fox.csv / bdg2_panther.csv / bdg2_rat_worst.csv | same corpus, built by `../prep_bdg2_subsets.py` (meter-selection study, paper Discussion) | MIT (source) | **yes** |

## bdg2.csv preprocessing specification

From the BDG2 hourly electricity meter table:
1. site **"Rat"** (the largest site in the corpus);
2. **15 buildings selected for minimal missingness** over the corpus period;
3. two years of hourly readings;
4. remaining gaps filled by an **ex-ante forward-fill then back-fill** rule, fixed before any
   modeling (no test-time information used).

The shipped `bdg2.csv` is the exact file used in all experiments (see `checksums.sha256`).
Honest provenance note: re-deriving a "15 least-missing at Rat" selection from today's
`electricity_cleaned.csv` reproduces only 5/15 of the shipped building set (the original
ranking differed in detail, e.g. in how zero readings were counted); the shipped, checksummed
file is the source of truth for the paper's results. The three **extension subsets** are, by
contrast, exactly reproducible: `prep_bdg2_subsets.py` builds `bdg2_fox.csv` /
`bdg2_panther.csv` (15 least-missing at sites Fox / Panther) and `bdg2_rat_worst.csv` (the 15
**most**-missing Rat meters with >=50% coverage --- the anti-selection subset for the
meter-selection-bias analysis) from the raw corpus table with the same fill rule.

## Loading convention

`online_eval.py:load_csv` drops the `date` column (and `rv1`/`rv2` for Appliances) and uses
all remaining numeric columns as channels. Each series is split by time into a pre-drift
warmup pool (first half) and a streamed test region (second half); z-normalization uses
warmup-pool statistics only.
