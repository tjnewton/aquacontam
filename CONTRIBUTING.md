# Contributing to AquaContam

Thank you for your interest in contributing to AquaContam! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository and clone your fork
2. Install in development mode: `pip install -e ".[dev,test]"`
3. Create a feature branch: `git checkout -b feat/your-feature`

## Development Workflow

### Code Style

- Python >= 3.10 with `X | Y` union syntax
- Formatter/Linter: `ruff` (line length 99)
- Docstrings: NumPy style
- Type hints required on all public API functions

```bash
# Format
ruff format src/ tests/

# Lint
ruff check src/ tests/

# Lint with auto-fix
ruff check --fix src/ tests/

# Type check
mypy src/aquacontam/
```

### Testing

All changes must pass the existing test suite:

```bash
# Run unit tests (fast, always run before submitting)
pytest tests/unit/ -x --tb=short -q

# Run with coverage
pytest tests/ --cov=aquacontam

# Integration tests (live-download probes; network required; excluded from CI)
pytest tests/integration/ -m slow
```

### Commit Messages

Use conventional commit format:

- `feat:` — new feature
- `fix:` — bug fix
- `refactor:` — code restructuring
- `test:` — test additions/changes
- `docs:` — documentation changes
- `ci:` — CI/CD changes

## Contributing Benchmark Results

### Submitting Model Results to the Leaderboard

1. Train your model using the AquaContam benchmark tasks
2. Use geographic stratification (EPA region splits) — results with random splits will not be accepted
3. Generate a submission JSON using the benchmark CLI or `scripts/reproduce.py`
4. Submit via pull request to the `submissions/` directory

### Adding New Data Sources

Data sources must:

- Implement the `DataSource` ABC from `src/aquacontam/data/base.py`
- Include schema validation
- Handle left-censored (non-detect) values correctly
- Never modify files in `data/raw/` — transformations write to `data/interim/` or `data/processed/`
- Preserve PWSID as 9-character strings (never cast to int)

### Adding New Models

Models must:

- Implement the `BaseModel` ABC from `src/aquacontam/models/base.py`
- Support both `predict()` and `predict_proba()` for classification
- Include unit tests

## Pull Request Process

1. Ensure all tests pass: `pytest tests/unit/`
2. Ensure linting passes: `ruff check src/ tests/`
3. Update documentation if adding new public API
4. Include a clear description of changes and their motivation
5. Reference any related issues

## Reporting Issues

Please use the GitHub issue templates for:

- **Bug reports**: Include steps to reproduce, expected behavior, and environment details
- **Feature requests**: Describe the use case and proposed solution
- **Data issues**: Include the data source, expected vs actual values

## Licensing of contributions

AquaContam's code is licensed under the Apache License 2.0, and the maintainer
retains the right to offer the project under additional or different license
terms. By submitting a contribution you certify the Developer Certificate of
Origin v1.1 (https://developercertificate.org/ — sign off each commit with
`git commit -s`) and you additionally grant the project maintainer a perpetual,
worldwide, non-exclusive, royalty-free right to license your contribution as
part of the project under any license approved by the Open Source Initiative.
Contributions that cannot carry this grant cannot be merged.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.
