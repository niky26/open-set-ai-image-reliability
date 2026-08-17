"""Validate the core saved result invariants used in the thesis."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_json(path):
    return json.loads(path.read_text())


phase2 = load_json(RESULTS / "closed_set" / "phase2_results.json")
phase3 = load_json(RESULTS / "closed_set" / "phase3_results.json")

assert np.isclose(phase2["accuracy"], phase3["accuracy"])
assert np.isclose(phase2["auroc"], phase3["auroc"])
assert np.isclose(phase3["optimal_temperature"], 1.2408854103475755)
assert phase3["high_conf_errors"] <= phase2["high_conf_errors"]

logo = pd.read_csv(
    RESULTS / "logo" / "experiment_2_logo_all7_combined_summary.csv"
)
assert len(logo) == 7
assert logo["heldout_generator"].nunique() == 7
assert (logo["test_n"] == 1166).all()
assert (logo["test_real_n"] == 583).all()
assert (logo["test_ai_n"] == 583).all()
assert np.allclose(logo["before_accuracy"], logo["after_accuracy"])
assert np.allclose(logo["before_auroc"], logo["after_auroc"])

external = pd.read_csv(RESULTS / "audit" / "external_overall_recomputed.csv")
raw = external.loc[external["variant"] == "raw"].iloc[0]
latest = external.loc[
    external["variant"] == "temperature_1.240885_latest"
].iloc[0]

assert int(raw[["tn", "fp", "fn", "tp"]].sum()) == 3000
assert int(raw["tn"] + raw["fp"]) == 500
assert int(raw["fn"] + raw["tp"]) == 2500
assert np.isclose(raw["accuracy"], latest["accuracy"])
assert np.isclose(raw["auroc"], latest["auroc"])
assert np.isclose(latest["temperature"], phase3["optimal_temperature"])

print("All saved-result checks passed.")
print(f"Closed-set accuracy: {phase3['accuracy']:.6f}")
print(f"Seven-fold LOGO mean accuracy: {logo['after_accuracy'].mean():.6f}")
print(f"External accuracy: {latest['accuracy']:.6f}")
