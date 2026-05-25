# Assumption Registry Row

Add one row to `research_ops/ASSUMPTION_REGISTRY.csv`.

```csv
id,priority,layer,assumption,current_evidence,risk_if_false,falsification_test,confidence_increases_if,confidence_collapses_if,required_artifacts,blocked_actions,status,next_diagnostic
```

## Field Guidance

- `id`: stable ID such as `A013`.
- `priority`: one of `P0`, `P1`, `P2`.
- `layer`: short snake_case area such as `execution`, `data_parity`, or
  `validation`.
- `status`: one of `open`, `partially_tested`, `supported`, `falsified`,
  `accepted_risk`, `blocked`.
- Evidence fields should cite files, reports, logs, or manifests when possible.
