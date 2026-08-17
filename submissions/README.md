# AquaContam Benchmark Submissions

Submit your model results to the AquaContam leaderboard by opening a PR that
adds a JSON file to this directory.

## Submission Format

Each submission is a JSON file named `<model_name>.json`:

```json
{
  "schema_version": "1.0",
  "model_name": "MyModel",
  "author": "Jane Doe",
  "institution": "University of Example",
  "paper_url": null,
  "code_url": "https://github.com/...",
  "date": "2026-02-17",
  "dataset_version": "1.0",
  "hardware": "1x NVIDIA A100",
  "results": [
    {
      "task": "T1",
      "analyte": "PFOS",
      "metrics": {
        "auroc": 0.92,
        "auprc": 0.85,
        "f1": 0.78
      }
    }
  ]
}
```

## Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Must be `"1.0"` |
| `model_name` | string | Unique model identifier |
| `results` | array | One or more task results |

## Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `author` | string | Author name(s) |
| `institution` | string | Affiliation |
| `paper_url` | string | Link to paper |
| `code_url` | string | Link to code repository |
| `date` | string | Submission date (YYYY-MM-DD) |
| `dataset_version` | string | Dataset version used |
| `hardware` | string | Hardware description |

## Valid Tasks and Metrics

| Task | Type | Primary Metric | Expected Metrics |
|------|------|----------------|------------------|
| T1 | Binary PFAS detection | auprc | auroc, auprc, f1 |
| T2 | Concentration regression | rmse | rmse, mae, r2 |
| T3 | Multi-PFAS profile | macro_auprc | macro_auroc, macro_auprc, macro_f1, hamming_loss, subset_accuracy |
| T4 | Heavy metal detection | auprc | auroc, auprc, f1 |
| T5 | Cross-contaminant transfer | auprc | auroc, auprc, f1 |
| T6 | Arsenic NGA transfer probe (public-supply to domestic wells) | auprc | auroc, auprc, f1 |
| T7 | Temporal UCMR3 to UCMR5 | auprc | auroc, auprc, f1 |

## Validation

Submissions are automatically validated on PR:

```bash
python -m aquacontam leaderboard validate submissions/my_model.json
```

## Ranking

The leaderboard is regenerated on every push to `main`:

```bash
python -m aquacontam leaderboard rank submissions/
python -m aquacontam leaderboard publish submissions/ -o LEADERBOARD.md
```
