# Assumption Registry Row

Add one row to `research_ops/ASSUMPTION_REGISTRY.csv`.

```csv
assumption_id,title,status,importance,fragility,falsification_risk,category,current_evidence,falsification_test,confidence_increases_if,confidence_destroyed_if,next_artifact,owner,last_updated
```

## Field Guidance

- `assumption_id`: stable ID such as `A011`.
- `status`: one of `open`, `partially_tested`, `supported`, `falsified`,
  `accepted_risk`, `blocked`.
- `importance`: one of `low`, `medium`, `high`, `critical`.
- `fragility`: one of `low`, `medium`, `medium_high`, `high`.
- `falsification_risk`: one of `low`, `medium`, `medium_high`, `high`.
- `category`: short snake_case category.
- Evidence fields should cite files, reports, logs, or manifests when possible.
