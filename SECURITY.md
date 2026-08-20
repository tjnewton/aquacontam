# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately via
[GitHub Security Advisories](https://github.com/tjnewton/aquacontam/security/advisories/new).
For non-sensitive issues (e.g., a dependency bump or a hardening suggestion), a regular
[issue](https://github.com/tjnewton/aquacontam/issues) is fine.

## Scope

AquaContam is a research benchmark: a Python library, a reproducibility pipeline, and a
frozen results archive. It runs locally, hosts no service, and handles no credentials or
personal data. The most security-relevant surfaces are the data-download stage (network
fetches verified against `data/checksums.sha256`) and model deserialization (weight files
are SHA-256-verified before `joblib.load()`).

## Supported versions

| Version | Supported |
|---------|-----------|
| 4.x     | Yes       |
| < 4.0   | No        |
