# Experiment 2 LOGO Results - Recovered Summary

The Colab runtime disconnected before the zip was downloaded, but the notebook output preserved the completed six remaining folds. These recovered metrics were combined with the earlier completed Midjourney fold.

## Completion Status

| Held-out generator | Status |
|---|---|
| Midjourney | Completed earlier |
| stable_diffusion_v_1_5 | Recovered from notebook output |
| glide | Recovered from notebook output |
| wukong | Recovered from notebook output |
| BigGAN | Recovered from notebook output |
| VQDM | Recovered from notebook output |
| ADM | Recovered from notebook output |

## Thesis Table

| Held-out Generator | Accuracy | AUROC | ECE Before | ECE After | Avg Conf Wrong Before | Avg Conf Wrong After | HCE Before | HCE After |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Midjourney | 0.8825 | 0.9408 | 0.0464 | 0.0398 | 0.8143 | 0.7987 | 52 | 44 |
| stable_diffusion_v_1_5 | 0.9220 | 0.9839 | 0.0331 | 0.0222 | 0.8054 | 0.7762 | 36 | 25 |
| glide | 0.9099 | 0.9918 | 0.0448 | 0.0296 | 0.8022 | 0.7669 | 40 | 29 |
| wukong | 0.9031 | 0.9727 | 0.0351 | 0.0141 | 0.7775 | 0.7467 | 33 | 28 |
| BigGAN | 0.9305 | 0.9991 | 0.0381 | 0.0328 | 0.8396 | 0.8096 | 35 | 28 |
| VQDM | 0.8962 | 0.9624 | 0.0403 | 0.0218 | 0.7965 | 0.7649 | 42 | 34 |
| ADM | 0.9082 | 0.9677 | 0.0379 | 0.0269 | 0.8087 | 0.7875 | 39 | 32 |
| Mean | 0.9075 | 0.9741 | 0.0394 | 0.0267 | 0.8063 | 0.7786 | 39.57 | 31.43 |
| SD | 0.0159 | 0.0197 | 0.0048 | 0.0084 | 0.0188 | 0.0216 | 6.29 | 6.27 |

HCE = high-confidence errors at confidence > 0.90.

## Main Interpretation

Across the seven LOGO folds, ResNet-50 achieved a mean accuracy of 0.9075 and mean AUROC of 0.9741 under unseen-generator shift. Temperature scaling preserved accuracy and AUROC because it does not alter the model decision boundary, but it improved reliability. Mean ECE decreased from 0.0394 to 0.0267, a 32.09% reduction. Mean high-confidence errors decreased from 39.57 to 31.43 per fold, a 20.58% reduction. These results support the thesis claim that post-hoc calibration can reduce overconfident errors under generator-level open-set conditions.

## Recovered Files

| File | Purpose |
|---|---|
| `outputs/experiment_2_remaining6_recovered_from_notebook.csv` | Six folds recovered from the notebook output |
| `outputs/experiment_2_logo_all7_combined_summary.csv` | Full detailed seven-fold combined table |
| `outputs/experiment_2_logo_all7_thesis_table.csv` | Compact thesis-ready table |
| `outputs/experiment_2_logo_all7_mean_std.csv` | Mean, SD, min, and max statistics |
| `outputs/recovered_experiment_2_plots/` | Recovered plot images embedded in the notebook |

## Caveat

The notebook preserved metrics and figures, but the disconnected runtime may have lost the saved `.keras` fold models and per-image prediction CSVs. This is acceptable for thesis reporting if only aggregate LOGO metrics and figures are needed.
