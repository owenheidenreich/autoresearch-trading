# Quote Age Truth Report

Decision: `require observability patch`

Diagnostic verdict: `unknown`

Reason: available logs do not contain enough broker/paper-submit trustworthy evidence

## Evidence

- Parsed rows: `6663`
- Files read: `15`
- Persisted quote-age rows: `1`
- Quote evidence rows: `1`
- Broker endpoint rows: `0`
- Trustworthy ratio among persisted: `0.0`
- Placeholder ratio among persisted: `0.0`

## Classification Counts

- `missing`: 6662
- `unreconstructable`: 1

## Conclusion

Existing inspected logs do not prove quote age truth unless the verdict is `pass`. Missing raw quote timestamps remain blocking evidence for paper-submit trust.
