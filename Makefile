# AquaContam paper reproducibility targets.
#
# The committed paper tables/numbers derive from a tracked, slimmed result
# snapshot (results/paper_frozen/). These targets regenerate and verify them.

PYTHON ?= python
FROZEN  ?= results/paper_frozen
SOURCE  ?= results_reframe
PF_SOURCE ?= results_provenance_free

.PHONY: help paper paper-tables paper-verify paper-freeze

help:
	@echo "paper        - regenerate tables from the frozen snapshot and verify consistency"
	@echo "paper-tables - regenerate paper/tables/ from \$$FROZEN ($(FROZEN))"
	@echo "paper-verify - run verify_paper.py against \$$FROZEN (hard-fails on table drift)"
	@echo "paper-freeze - rebuild the frozen snapshot from \$$SOURCE ($(SOURCE)),"
	@echo "               folding in the env-only run from \$$PF_SOURCE ($(PF_SOURCE))"

## Regenerate the tables from the frozen snapshot, then verify consistency.
## (Figures that embed per-row arrays -- ROC/PR, risk map -- need the full
## SOURCE run, not the slimmed snapshot; regenerate those separately.)
paper: paper-tables paper-verify

paper-tables:
	PYTHONUTF8=1 $(PYTHON) paper/generate_tables.py --results $(FROZEN) --output paper/tables

paper-verify:
	@test -f paper/skeleton.md || { \
		echo "manuscript sources not distributed (see README); skipping paper-verify"; \
		exit 0; }
	PYTHONUTF8=1 $(PYTHON) paper/verify_paper.py --results-dir $(FROZEN)

## Rebuild the slim frozen snapshot (drops large arrays, copies ancillary JSONs,
## folds in the provenance-free regime + t6 extra, writes checksums + MANIFEST)
## from completed pipeline runs.
paper-freeze:
	@test -d $(SOURCE) || { \
		echo "paper-freeze is maintainer-only: source tree '$(SOURCE)' not present"; \
		echo "(regime result trees are not included in the public release)"; \
		exit 1; }
	$(PYTHON) paper/freeze_results.py --source $(SOURCE) --frozen $(FROZEN) \
		--provenance-free-source $(PF_SOURCE) --extra results/t6_arsenic.json
