# Path-D corrected-v3.2 release supersession

Date: 2026-08-03

Status: `SUPERSEDED_INVALID_EXPERIMENT`

The corrected-v3.2 Claude release is preserved byte-for-byte as historical
evidence. It no longer authorizes work and its current-source reconciliation is
not a valid precondition for invariant-only legacy utilities.

The release governed the later `signed18_model_side_nearest` confirmation. The
offline runtime decision-parity audit established that all 445,063 fitted rows
used ThetaData SPX context unavailable at the declared decision clock. The
immutable model disposition is
`INVALID_EXPERIMENT_AND_NOT_RUNTIME_DECISION_PARITY_ELIGIBLE`.

Authoritative bindings:

- immutable v3.2 Claude release SHA-256:
  `17c028d56b5473357717abac598057f6fdf183b41905420e13286a930cc3d6a7`;
- decision-parity result SHA-256:
  `6b4d3fd66cf575aa7db4e1f51bd1452baf42a33018b34e121ab43df019fc0e56`;
- machine-readable supersession marker:
  `v4/audit/autoresearch/pathd_v32_release_supersession_2026_08_03.json`;
- marker validator:
  `v4/research/pathd_entry_dataset.py::assert_v32_release_superseded`.

The marker authorizes no v3.2 fit, corpus decode, evidence opening, protected
holdout access, broker/paper activity, promotion, or default change. Those
legacy execution paths remain fail-closed with an explicit superseded status.
The only behavior retired is the false requirement that current source bytes
must continue to match a release governing an invalidated result.

The successor is the distinct Path-D Phase-1 causal rebuild. Supersession does
not certify that successor's economic edge and does not open its holdout.
