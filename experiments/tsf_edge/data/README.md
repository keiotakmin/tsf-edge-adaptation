# Datasets

Six public multivariate series. `get_data.sh` downloads the five that are not shipped and
verifies them against `checksums.sha256` — the checksums of the exact files used in the paper.

| File | Source | License | Shipped? |
|---|---|---|---|
| ETTh1/ETTh2/ETTm1/ETTm2.csv | [ETDataset](https://github.com/zhouhaoyi/ETDataset) (ETT-small) | see source repo | no (downloaded) |
| appliances.csv | [UCI Appliances energy prediction](https://archive.ics.uci.edu/dataset/374) (`energydata_complete.csv`, renamed) | CC BY 4.0 | no (downloaded) |
| bdg2.csv | derived from [Building Data Genome 2](https://github.com/buds-lab/building-data-genome-project-2) (Zenodo DOI [10.5281/zenodo.3887306](https://doi.org/10.5281/zenodo.3887306)) | MIT (source) | **yes** |

## bdg2.csv preprocessing specification

From the BDG2 hourly electricity meter table:
1. site **"Rat"** (the largest site in the corpus);
2. the **15 buildings with the least missingness** over the corpus period;
3. two years of hourly readings;
4. remaining gaps filled by an **ex-ante forward-fill then back-fill** rule, fixed before any
   modeling (no test-time information used).

The shipped `bdg2.csv` is the exact file used in all experiments (see `checksums.sha256`);
the specification above documents how it was derived from the public corpus.

## Loading convention

`online_eval.py:load_csv` drops the `date` column (and `rv1`/`rv2` for Appliances) and uses
all remaining numeric columns as channels. Each series is split by time into a pre-drift
warmup pool (first half) and a streamed test region (second half); z-normalization uses
warmup-pool statistics only.
