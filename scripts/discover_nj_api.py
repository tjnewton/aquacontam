"""NJ DEP Drinking Water Viewer API discovery script.

Demonstrates the waterviewer.nj.gov REST API endpoints discovered via
Playwright + manual probing. Run this to verify the API is still accessible
and to explore available data.

Usage:
    python scripts/discover_nj_api.py

Findings:
- Auth: GET https://waterviewer.nj.gov/ to get XSRF-TOKEN cookie,
  then send X-XSRF-TOKEN header on all requests.
- Inventory: /sdwis/DashMain — water system list (OData, paginated)
- Facilities: /sdwis/FacilityList — facility coords (per-system via n0=)
- Analytes: /LookUp/AnalyteCodes — all analyte codes incl. 30 PFAS
- Samples: /sdwis/SamplesSearchResults — reCAPTCHA-protected, returns empty
"""

from __future__ import annotations

import requests


def main() -> None:
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    # Authenticate
    resp = session.get("https://waterviewer.nj.gov/", timeout=30)
    resp.raise_for_status()
    xsrf = session.cookies.get("XSRF-TOKEN", "")
    session.headers["X-XSRF-TOKEN"] = xsrf
    print(f"[OK] Session established (XSRF token: {len(xsrf)} chars)")

    # 1. Water system inventory
    print("\n[1] Water system inventory (/sdwis/DashMain) ...")
    resp = session.get(
        "https://waterviewer.nj.gov/sdwis/DashMain?skip=0&take=5",
        timeout=30,
    )
    data = resp.json()
    systems = data.get("value", [])
    print(f"  Sample of {len(systems)} records (paginate with skip/take):")
    for s in systems[:3]:
        n0 = str(s.get("NUMBER0", "")).strip()
        name = s.get("NAME", "")
        county = s.get("D_PRIN_CNTY_SVD_NM", "")
        status = s.get("ACTIVITY_STATUS_CD", "")
        print(f"    {n0}: {name} ({county}, status={status})")

    # 2. Facility coordinates
    if systems:
        n0 = str(systems[0]["NUMBER0"]).strip()
        is_num = systems[0]["TINWSYS_IS_NUMBER"]
        print(f"\n[2] Facility coordinates for {n0} (/sdwis/FacilityList) ...")
        filt = f"(TINWSYS_IS_NUMBER eq {is_num} and TINWSYS_ST_CODE eq 'NJ')"
        resp = session.get(
            f"https://waterviewer.nj.gov/sdwis/FacilityList?n0={n0}&$filter={filt}",
            timeout=30,
        )
        facs = resp.json().get("value", [])
        print(f"  Found {len(facs)} facilities")
        for f in facs[:3]:
            lat = f.get("LATITUDE_MEASURE")
            lon = f.get("LONGITUDE_MEASURE")
            ftype = f.get("TYPE_CODE", "").strip()
            print(f"    {f.get('ST_ASGN_IDENT_CD', '').strip()}: ({lat}, {lon}) type={ftype}")

    # 3. Analyte codes
    print("\n[3] PFAS analyte codes (/LookUp/AnalyteCodes) ...")
    resp = session.get("https://waterviewer.nj.gov/LookUp/AnalyteCodes", timeout=30)
    analytes = resp.json()
    pfas = [
        a
        for a in analytes
        if any(
            x in str(a.get("text", "")).upper()
            for x in ["PERFLUOR", "PF", "HFPO", "FLUOROTELOMER"]
        )
    ]
    print(f"  {len(pfas)} PFAS codes out of {len(analytes)} total:")
    for a in pfas[:10]:
        print(f"    {a['code']}: {a['text']}")
    if len(pfas) > 10:
        print(f"    ... and {len(pfas) - 10} more")

    # 4. Samples (reCAPTCHA-protected)
    print("\n[4] Sample data (/sdwis/SamplesSearchResults) ...")
    resp = session.get(
        "https://waterviewer.nj.gov/sdwis/SamplesSearchResults?skip=0&take=5",
        timeout=30,
    )
    data = resp.json()
    count = len(data.get("value", []))
    print(f"  Records returned: {count}")
    if count == 0:
        print("  (Expected: 0 — endpoint requires reCAPTCHA for sample data)")

    # 5. Analyte groups
    print("\n[5] PFAS analyte groups (/LookUp/AnalyteGroups) ...")
    resp = session.get("https://waterviewer.nj.gov/LookUp/AnalyteGroups", timeout=30)
    groups = resp.json()
    pfas_groups = [g for g in groups if "PF" in str(g.get("code", "")).upper()]
    for g in pfas_groups:
        print(f"    {g['code']}: {g['text']}")

    print("\n[OK] Discovery complete.")


if __name__ == "__main__":
    main()
