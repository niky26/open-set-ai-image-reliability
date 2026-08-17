# Thesis Result Audit

## Authoritative closed-set run

- Phase 1 accuracy: 0.572225
- Phase 2 accuracy: 0.957137
- Phase 3 accuracy: 0.957137
- Phase 3 temperature: 1.240885410
- Phase 2 to 3 ECE reduction: 7.98%
- Phase 2 to 3 HCE count reduction: 21.15%

## True seven-fold LOGO

- Mean accuracy: 0.907498
- Mean AUROC: 0.974058
- Mean ECE: 0.039381 -> 0.026743
- ECE reduction: 32.09%
- Mean HCE count: 39.57 -> 31.43

## External Defactify sample

- Raw accuracy/AUROC: 0.811667 / 0.792387
- Saved old-temperature ECE-15: 0.091314
- Recomputed latest-temperature ECE-15: 0.120104
- Recomputed latest-temperature HCE: 283 of 2316 high-confidence predictions
- Recomputed latest-temperature HCE rate among high-confidence predictions: 0.122193
- Recomputed latest-temperature HCE rate over all images: 0.094333

## Metric-definition note

The latest closed-set JSON uses positive-class ECE with 10 bins. The true LOGO and Defactify notebooks use top-label confidence ECE. Absolute ECE values across these protocols should not be compared unless per-image predictions are recomputed using one estimator. Before/after comparisons within each protocol remain valid.