#!/usr/bin/env python
"""R7 M5: add missing prior-art / lineage citations + first-appearance renumber.

Adds six references the R7 referee flagged as missing (Peters 2016 ICP-name lineage;
Ganin/Arjovsky/Sagawa domain-adversarial / IRM / Group-DRO lineage; Wadoux 2021 spatial-CV
counter-view; Martinez-Morata & Nigra 2022 As/U racial-disparity prior art) and renumbers
the whole bibliography to ascending order of first appearance (Nature style, enforced by the
G2 `check_bibliography_bijection` linter). Reuses the proven helpers from the #55
`renumber_references.py`; that script is a spent one-shot (its edits are already applied), so
this is a parallel R7 pass over the current skeleton. Dry-run by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from renumber_references import (  # noqa: E402
    TOKEN_RE,
    expand_token,
    parse_bibliography,
    split_sections,
    wrap_entry,
)

SKELETON = Path(__file__).resolve().parent / "skeleton.md"

# --- citation-site edits (unique-substring anchors in the current body) ------
SITE_EDITS: list[tuple[str, str]] = [
    # ICP name collision: distinguish from Invariant Causal Prediction (Peters 2016).
    (
        "An Invariant Contamination Predictor (ICP) retains all features but learns",
        "An Invariant Contamination Predictor (ICP), named by analogy to invariance-based "
        "learning and distinct from the Invariant Causal Prediction framework^PETERS^, "
        "retains all features but learns",
    ),
    # GRL / domain-adversarial lineage (Ganin 2016).
    (
        "The gradient reversal layer (GRL) negates adversary",
        "The gradient reversal layer (GRL)^GANIN^ negates adversary",
    ),
    # Invariance objective lineage (Arjovsky 2019 / IRM).
    (
        "from which monitoring intensity cannot be predicted.",
        "from which monitoring intensity cannot be predicted, an objective related to "
        "invariant risk minimization^ARJOVSKY^.",
    ),
    # Group DRO lineage (Sagawa 2020).
    (
        "propensity scores and Group DRO with",
        "propensity scores and Group DRO^SAGAWA^ with",
    ),
    # Spatial-CV counter-view (Wadoux 2021).
    (
        "chose to monitor them.",
        "chose to monitor them. Spatial cross-validation can be conservative for "
        "probability-sample designs^WADOUX^, but the monitoring process here is precisely "
        "the non-random design we diagnose, so leakage-aware evaluation remains appropriate.",
    ),
    # Nearest equity prior art (Martinez-Morata & Nigra 2022).
    (
        "sociodemographic disparities in PFAS exposure and pollution burden^36,37,38^.",
        "sociodemographic disparities in PFAS exposure and pollution burden^36,37,38^ and in "
        "public drinking-water arsenic and uranium^MARTINEZ^.",
    ),
]

# --- new bibliography entries (Nature house style) ---------------------------
NEW_ENTRIES: dict[str, str] = {
    "PETERS": (
        "Peters, J., Bühlmann, P. & Meinshausen, N. Causal inference by using invariant "
        "prediction: identification and confidence intervals. *J. R. Stat. Soc. B* **78**, "
        "947--1012 (2016)."
    ),
    "GANIN": (
        "Ganin, Y. et al. Domain-adversarial training of neural networks. *J. Mach. Learn. "
        "Res.* **17**, 1--35 (2016)."
    ),
    "ARJOVSKY": (
        "Arjovsky, M., Bottou, L., Gulrajani, I. & Lopez-Paz, D. Invariant risk "
        "minimization. Preprint at https://arxiv.org/abs/1907.02893 (2019)."
    ),
    "SAGAWA": (
        "Sagawa, S., Koh, P. W., Hashimoto, T. B. & Liang, P. Distributionally robust neural "
        "networks for group shifts: on the importance of regularization for worst-case "
        "generalization. in *Proc. ICLR* (2020)."
    ),
    "WADOUX": (
        "Wadoux, A. M. J.-C., Heuvelink, G. B. M., de Bruin, S. & Brus, D. J. Spatial "
        "cross-validation is not the right way to evaluate map accuracy. *Ecol. Modell.* "
        "**457**, 109692 (2021)."
    ),
    "MARTINEZ": (
        "Martinez-Morata, I. et al. Nationwide geospatial analysis of county racial and "
        "ethnic composition and public drinking water arsenic and uranium. *Nat. Commun.* "
        "**13**, 7461 (2022)."
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="rewrite skeleton.md in place")
    args = ap.parse_args()

    text = SKELETON.read_text(encoding="utf-8")
    before, refs_block, after = split_sections(text)

    for old, new in SITE_EDITS:
        n = before.count(old)
        if n != 1:
            raise SystemExit(f"site-edit anchor not unique (count={n}): {old!r}")
        before = before.replace(old, new)

    bib = parse_bibliography(refs_block)
    for k in NEW_ENTRIES:
        if k in bib:
            raise SystemExit(f"new-entry key collides with existing: {k}")
    bib.update(NEW_ENTRIES)

    order: list[str] = []
    seen: set[str] = set()
    for m in TOKEN_RE.finditer(before):
        for key in expand_token(m.group(1)):
            if key not in seen:
                seen.add(key)
                order.append(key)

    cited, have = set(order), set(bib)
    if cited - have:
        raise SystemExit(f"cited but no entry: {sorted(cited - have)}")
    if have - cited:
        raise SystemExit(f"orphan entry (never cited): {sorted(have - cited)}")

    old_to_new = {key: i + 1 for i, key in enumerate(order)}

    def remap(m):
        nums = sorted(old_to_new[k] for k in expand_token(m.group(1)))
        return "^" + ",".join(str(n) for n in nums) + "^"

    new_body = TOKEN_RE.sub(remap, before)
    lines = ["## References", ""]
    for key in order:
        lines.append(wrap_entry(old_to_new[key], bib[key]))
        lines.append("")
    new_refs = "\n".join(lines).rstrip() + "\n\n"
    new_text = new_body + new_refs + after

    print(f"references: {len(order)} (was {len(parse_bibliography(refs_block))})")
    inv = {v: k for k, v in old_to_new.items()}
    for new_num in range(1, len(order) + 1):
        key = inv[new_num]
        label = "NEW " if key in NEW_ENTRIES else "  "
        if key in NEW_ENTRIES or not key.isdigit() or int(key) != new_num:
            print(f"  {label}{key:>9} -> {new_num}")
    appearances = [old_to_new[k] for k in order]
    assert appearances == sorted(appearances), "first-appearance not ascending"

    if args.apply:
        SKELETON.write_text(new_text, encoding="utf-8")
        print(f"\nWROTE {SKELETON}")
    else:
        print("\nDRY RUN — nothing written (pass --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
