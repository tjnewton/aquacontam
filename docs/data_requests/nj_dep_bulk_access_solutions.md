# NJ DEP PFAS Sample Data — Access Provenance

New Jersey DEP drinking-water chemical-sample results are public records under the
New Jersey Open Public Records Act (OPRA). The WaterViewer portal
(`waterviewer.nj.gov`) presents them through an interactive lookup rather than a bulk
download, so the OPRA request was answered with a pointer to that portal (see
`nj_dep_opra_response.md`).

## How the data were collected

The records were retrieved by reading the same public WaterViewer pages a person
would, using a headed browser session and only that session's own XSRF token from
those pages — no account, no credentials, and no CAPTCHA solving or access
circumvention. The public inventory and facility-coordinate endpoints are read
directly; the public sample-results endpoint is paginated from within the ordinary
browser session. Implementation: `scripts/scrape_nj_dep.py`; the parsed output drops
into `NjDepSource._parse_manual_export()`. Collection is human-launched,
checkpoint/resumable, and rate-limited.

## Alternatives considered (for reproducers)

- **UCMR5 already covers ~97.7% of the NJ population by served count** (266 NJ systems),
  and all 25 NJ DEP PFAS analytes are a subset of UCMR5's — so for the benchmark the
  UCMR5 NJ data are largely sufficient; the WaterViewer records add small-system coverage.
- The NJDEP Open Data / ArcGIS Hub hosts only industrial-discharge and groundwater-permit
  PFAS datasets, not drinking-water monitoring.
- The EHP 2024 NJ PFAS study (doi:10.1289/EHP12787) used the same source via records
  request / direct contact with the NJDEP Division of Science and Research.
