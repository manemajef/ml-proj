# Submission validation reference

`desired_submission.csv` is the protected prediction vector used by
`scripts/validate_submission.py`. The notebook writes to
`data/Group_27_Submission.csv`, never to this file.

## Provenance

- SHA-256: `41a8bd3fd0098ae957e96ab5a787cbbbe1bf3fd529b6ba5b58142f78fc3d5104`
- Created from a clean isolated notebook refit on 2026-07-16.
- The refit dynamically selected XGBoost depth 6, learning rate 0.03, 700
  trees, and the `Stronger` regularization profile.
- The final feature set contained ISO week and the continuous time index.
- A second cached isolated run reproduced the file byte-for-byte.
- It contains 15,866 unique test IDs in official test order, no missing values,
  and finite probabilities in `[0, 1]`.

This establishes the exact prediction vector chosen for submission. It does
not establish its hidden-test AUC. The independently scored v2 artifact remains
`data/predictions/predictions-889.csv`; it is intentionally a different vector.

Do not replace the protected reference merely to make a failing validation
pass. A replacement requires an explicit model decision, a fresh isolated
refit, integrity checks, and an intentional SHA-256 update in the validator.
