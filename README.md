# Open-Set Reliability and Confidence Calibration in AI-Generated Image Detection

Reproducibility package for a ResNet-50 study of AI-generated image detection under familiar-source, held-out-generator, and recent cross-dataset evaluation.

## Research questions

**Main question:** How does open-set generator shift affect the reliability of AI-generated image detectors in cybersecurity deployment scenarios?

**Calibration sub-question:** Can post-hoc temperature scaling improve confidence calibration and reduce high-confidence errors under distribution shift without affecting classification accuracy?

## Repository structure

```text
.
├── notebooks/
│   ├── 01_closed_set_resnet50.ipynb
│   ├── 02_logo_resnet50.ipynb
│   ├── 03_external_defactify_stress_test.ipynb
│   └── archive/external_eval_original.ipynb
├── scripts/
│   └── logo_resnet50_colab.py
├── results/
│   ├── closed_set/
│   ├── logo/
│   ├── external/
│   └── audit/
├── docs/thesis.docx
├── requirements.txt
└── .gitignore
```

## Datasets

The datasets are downloaded at runtime and are not committed.

1. **Primary dataset:** [Unbiased Tiny GenImage](https://www.kaggle.com/datasets/cartografia/unbiased-tiny-genimage)
   - Kaggle identifier: `cartografia/unbiased-tiny-genimage`
   - 5,828 Nature images and seven AI-generator folders with 2,500 images each
2. **External dataset:** [Defactify Image Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset)
   - Hugging Face identifier: `Rajarshi-Roy-research/Defactify_Image_Dataset`
   - Saved evaluation sample: 500 real MS COCO images and 500 images from each of five AI sources

Dataset licenses and terms remain those of the original publishers.

## Experiments

### 1. Closed-set reference

`notebooks/01_closed_set_resnet50.ipynb` trains and calibrates the familiar-source ResNet-50 reference.

- Phase 1: ImageNet-initialized baseline
- Phase 2: task-specific fine-tuning
- Phase 3: validation-only scalar temperature scaling

Authoritative artifacts are in `results/closed_set/`.

### 2. Seven-fold LOGO

`notebooks/02_logo_resnet50.ipynb` implements leave-one-generator-out evaluation. For every fold, the held-out generator is absent from training and validation, a fresh ResNet-50 is trained, and temperature is fitted using validation logits only.

The saved notebook was used to complete the six remaining folds after Midjourney had finished separately. To run all seven folds from scratch, set:

```python
RUN_FOLDS = AI_GENERATORS
```

Do not remove Midjourney from `AI_GENERATORS`; it must be available as a seen source whenever another generator is held out.

### 3. Recent external stress test

`notebooks/03_external_defactify_stress_test.ipynb` evaluates the saved closed-set model on Defactify. It preserves per-image predictions, class-conditional errors, confidence measures, and environment metadata.

The thesis-authoritative temperature is:

```python
TEMPERATURE = 1.2408854103475755
```

The notebook does not fit temperature on external labels.

## Verified result summary

| Condition | Accuracy | AUROC | ECE before | ECE after |
|---|---:|---:|---:|---:|
| Closed-set fine-tuned | 0.9571 | 0.9921 | 0.0344 | 0.0316 |
| Seven-fold LOGO mean | 0.9075 | 0.9741 | 0.0394 | 0.0267 |
| External sample | 0.8117 | 0.7924 | 0.1357 | 0.1201 |

Additional verified findings:

- Closed-set high-confidence errors: 52 to 41 after scaling.
- LOGO mean high-confidence errors: 39.57 to 31.43.
- External confusion counts: TN 233, FP 267, FN 298, TP 2,202.
- The external real-image false-positive rate is 267/500 = 0.534.
- Positive temperature scaling leaves accuracy, AUROC, and the 0.5-threshold confusion matrix unchanged.

See `results/audit/thesis_result_audit.md` for the complete cross-check.

## External-temperature artifact note

The files directly under `results/external/` preserve the original external notebook run, which used the previously saved temperature `1.7894`. The thesis-authoritative recalculation using `1.2408854103475755` is stored in:

- `results/audit/external_overall_recomputed.csv`
- `results/audit/external_generator_recomputed_latest_temperature.csv`
- `results/audit/thesis_result_audit.json`

This distinction is intentional and should be retained when citing results.

## Running locally

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Open the notebooks in numerical order. A CUDA-capable GPU is strongly recommended for training.

Validate the committed result artifacts without retraining:

```bash
python scripts/verify_results.py
```

The primary notebook downloads the Kaggle dataset through `kagglehub`. Kaggle authentication may be required depending on the local environment. The external notebook downloads Defactify through Hugging Face `datasets`.

## Model file

The trained `.keras` model is intentionally excluded because it is approximately 270 MB. Place it locally at the path configured by the external notebook, for example:

```text
models/resnet50_finetuned.keras
```

If the model must be versioned, use [Git LFS](https://git-lfs.com/):

```bash
git lfs install
git lfs track "*.keras"
```

## Metric definitions

- Binary decision threshold: `0.5`
- High-confidence threshold: top-label confidence at least `0.90`
- Closed-set JSON: positive-class ECE with 10 bins
- LOGO: top-label ECE with 10 bins
- External audit: top-label ECE with 10 and 15 bins, plus positive-class ECE for comparison

Absolute ECE values from unlike estimators should not be pooled. Before/after comparisons remain valid within each protocol.

## Cybersecurity interpretation

The detector is evaluated as a triage signal, not proof of image provenance. Under the external sample, 267 of 500 real images were flagged as AI-generated. Accordingly, deployment should include human review, reversible actions, provenance checks, logging, and drift monitoring rather than autonomous denial.

## Committing to GitHub

```bash
git init
git add .
git commit -m "Add thesis reproducibility package"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```
