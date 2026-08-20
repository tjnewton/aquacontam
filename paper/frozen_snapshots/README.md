# Frozen snapshots

Committed snapshots of reader-facing numbers that are **not** stored in the frozen
analysis archive (`results/paper_frozen/`) because they derive from the git-ignored
`data/interim/merged_wq.parquet` (dataset totals, per-source record counts) or from the
hand-maintained Table 1 (analyte counts, censoring rates).

These files live **outside** `results/paper_frozen/` on purpose: the frozen archive must
stay byte-unchanged (gate clause G6), but `paper_numbers.py`'s `ds_*` markers still need a
committed, CI-safe source. Re-pointing the `ds_*` loaders here removes the parquet
dependency, so `paper_numbers.py --check` no longer silently skips those markers when the
parquet is absent (e.g. in CI).

## `dataset_snapshot.json`

Totals + per-source Table-1 numbers. The `totals` and parquet-backed `sources.*.records`
were verified byte-equal to the parquet on 2026-07-01 with:

```bash
PYTHONUTF8=1 python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/interim/merged_wq.parquet", columns=["source","pwsid","latitude"])
print("samples", len(df))
print("systems", df["pwsid"].nunique())
print("geocoded", df[df["latitude"].notna()]["pwsid"].nunique())
print(df["source"].value_counts().to_dict())
PY
```

`oh_epa` (26,554) is excluded from the merged parquet (zero unique PWSIDs after dedup) but
its record count is reader-facing in Table 1, so it is carried here as metadata.

`sources.sdwis.records = 916,899` is the keep-parse count, and the frozen analysis JSONs
are computed on this same 916,899-row keep-parse (the canonical re-freeze regenerated the
entire archive from one pinned DAG run). The earlier 694,419-row legacy-parse figure is
historical provenance only and describes no current analysis.

To re-verify equality at any time, re-run the command above and compare to this file.
