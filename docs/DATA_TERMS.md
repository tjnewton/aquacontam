# AquaContam data sources — attributions, terms, and redistribution status (P2/M9)

The AquaContam **code** is licensed under the Apache License 2.0. The **compiled dataset** is a derived
compilation of the sources below; it is distributed under **CC BY 4.0 where source terms
permit**, and every source retains its original terms. This manifest is the per-source
attribution/terms record the compilation license points to (referenced from the paper's
Data Availability statement and the Zenodo deposit).

| Source | Provider | Access route | Terms / license | Redistribution in the compiled dataset |
|---|---|---|---|---|
| UCMR5, UCMR3 | U.S. EPA | Public bulk download | U.S. Government work (public domain) | Yes |
| SDWIS (Lead & Copper Rule) | U.S. EPA | Public bulk download (SDWA datasets) | U.S. Government work (public domain) | Yes |
| Water Quality Portal (WQP) | USGS / EPA | Public API | U.S. Government work (public domain) | Yes |
| EPA FRS, TRI, EJScreen | U.S. EPA | Public bulk download | U.S. Government work (public domain) | Yes (as derived features) |
| DoD PFAS sites | U.S. DoD | Public download | U.S. Government work (public domain) | Yes (as derived features) |
| NLCD 2021 | USGS (MRLC) | Public raster download | Public domain | Yes (as derived zonal features) |
| USGS Principal Aquifers | USGS | Public download | Public domain | Yes (as derived features) |
| TIGER block groups | U.S. Census Bureau | Public download | Public domain | Yes (as derived apportionment only) |
| Michigan MPART | State of Michigan (EGLE) | Public ArcGIS service | Public state data | Yes |
| California GeoTracker GAMA | California SWRCB | Public CSV download | Public state data | Yes |
| Washington DOH | Washington State DOH | Public CSV download | Public state data | Yes |
| Missouri DNR | Missouri DNR | Public ArcGIS service | Public state data | Yes |
| Ohio EPA | Ohio EPA | Public ArcGIS service | Public state data | Excluded from the merged dataset (PWSID overlap with SDWIS; detected-only reporting) |
| North Carolina DEQ | North Carolina DEQ | No bulk download | Public state data | Summary use only |
| New Jersey DEP | New Jersey DEP (WaterViewer) | Headed-browser collection (portal sits behind a bot-protection layer; described in Data Availability) | Public state records | Yes (records are public; access route is not self-serve) |
| Minnesota MDH | Minnesota Dept. of Health | Email request (`docs/data_requests/`) | Provided on request | **Pending confirmation**; until the agency confirms, the MDH records are excluded from the public bundle |
| NGA arsenic (T6) | USGS (DOI 10.5066/P9JMUAPY) | Public download | Public domain, cite DOI | Yes |

Notes

- "Public domain" federal rows follow 17 U.S.C. § 105 (works of the U.S. Government).
- The compiled dataset adds harmonization, geocoding, and derived features; CC BY 4.0
  applies to that compilation layer, not to the underlying public-domain facts.
- Any source whose terms are later found to conflict is removed from the public bundle
  and listed here with the change dated.
