# Third-party components

## TabPFN v2 (Prior Labs GmbH) — foundation-model baseline

The TabPFN v2 checkpoint is **not** redistributed in this repository or in the
Zenodo archive; the `tabpfn` package downloads it from the upstream provider
on first use. The TabPFN code and the v2 model weights are licensed under the
Prior Labs License (Apache 2.0 with an additional attribution provision):

  https://github.com/PriorLabs/TabPFN/blob/49394b0/LICENSE

Built with PriorLabs-TabPFN.

Note: this project pins `tabpfn<7` and the explicit v2 checkpoint. Newer
`tabpfn` releases default to TabPFN-3 weights, which carry a different,
non-commercial license.

## US Census Bureau cartographic boundary file

`paper/assets/us_states_cb2023_5m.geojson` is a work of the United States
Government (public domain; 17 U.S.C. 105). See `paper/assets/README.md`.

## Water-quality and auxiliary data sources

Per-source attributions, terms, and redistribution status are listed in
`docs/DATA_TERMS.md`.
