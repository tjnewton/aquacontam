# Minnesota MDH PFAS — Bulk Data Export Provenance

**Status**: Fulfilled — bulk export received
**Provider**: Minnesota Department of Health (MDH), Drinking Water Protection
**Contact**: health.drinkingwater@state.mn.us
**Received**: 2026-04-23
**Staged**: `data/raw/mn_mdh_pfas.xlsx` (the agency-provided workbook)
**SHA-256**: `c937216263943cdc566ba7814708b7e7de24f498567ea6c1fe1d0367af6520c5` (pinned in `data/checksums.sha256`)

---

## Summary

MDH does not publish bulk PFAS drinking-water monitoring data through a public API or
download. The public-facing data is the interactive **PFAS in Minnesota's Public Water
Systems** dashboard (`https://www.health.state.mn.us/communities/environment/water/pfasmap.html`),
which is a lookup tool, not a bulk export. A complete machine-readable export was provided
on request as a single-sheet Excel workbook — the same manual-export pattern used for the
NJ DEP source (see `nj_dep_opra_response.md`).

## Contents (as received)

- **410,363 rows** → **247,230** after collapsing the redundant `ANALYTE_GROUP` panel
  replication (each measurement is repeated across `PFAS_533` / `PFC_EXPANDED` /
  `PFAS_Regulated`).
- **1,357 public water systems** (878 Community + 475 Nontransient-Noncommunity + 1
  Transient-Noncommunity).
- **27 PFAS analytes**, including all six EPA-regulated compounds (PFOS, PFOA, PFHxS,
  PFNA, PFBS, HFPO-DA).
- Collection dates **2005–2026**.
- Mixed reporting units (`ng/L` and `ug/L`), harmonized to µg/L on load.
- ~79% of rows are substitution-at-the-reporting-limit non-detects (`RESULT == REPORTING_LIMIT`).

## Columns

`PWS_ID` (7-digit int; federal id = `"MN"` + 7 digits), `PWS_TYPE`, `SAMPLE_ID`,
`ANALYTE_GROUP`, `ANALYTE` (full chemical name + parenthetical abbreviation), `RESULT`,
`UNIT_MEASURE_CODE`, `RESULT_CODE` (unused — all null), `REPORTING_LIMIT`, `COLLECTION_DATE`.

## Parsing

`src/aquacontam/data/mn_mdh.py` (`MnMdhSource`): collapses the `ANALYTE_GROUP` replication
(keeping the most-sensitive reading per `(PWS_ID, SAMPLE_ID, ANALYTE, COLLECTION_DATE)`),
maps the 27 full-chemical-name analytes to the project's canonical short codes, harmonizes
units per row, and marks `RESULT <= REPORTING_LIMIT` as left-censored. There are no
coordinates in the export; systems are geocoded by PWSID in preprocessing (99.8% are
present in the national SDWIS table with a ZIP code).

## Reproducibility

This is a manually-provided bulk export (no public API), analogous to the NJ DEP source.
The parsed `mn_mdh.parquet` is excluded from the project's public dataset release pending
confirmation from MDH that redistribution is permitted; reproducers should request the bulk
export from MDH at the contact above and parse it locally with the documented loader
(`MnMdhSource`). The raw workbook is excluded from git (`data/raw` is gitignored); its
SHA-256 is pinned in `data/checksums.sha256` so a requester can verify they received the
same export.
