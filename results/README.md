# Result artifacts

## `closed_set/`

- `phase1_results.json`: ImageNet-initialized reference
- `phase2_results.json`: fine-tuned detector
- `phase3_results.json`: final validation-temperature-scaled detector

## `logo/`

- `experiment_2_logo_all7_combined_summary.csv`: all seven held-out-generator folds
- `experiment_2_logo_all7_mean_std.csv`: aggregate descriptive statistics
- `experiment_2_logo_all7_thesis_table.csv`: compact reporting table
- `midjourney/`: raw saved artifacts for the separately completed Midjourney fold
- `figures/`: recovered plots from the remaining folds

## `external/`

Original Defactify evaluation artifacts, including the 3,000-row prediction CSV. These preserve the earlier saved temperature `1.7894`.

## `audit/`

Authoritative thesis audit and external recalculation using the final Phase 3 temperature `1.2408854103475755`.

Do not average ECE values that use different confidence definitions or bin counts.
