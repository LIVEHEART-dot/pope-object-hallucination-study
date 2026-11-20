"""Validate all three prompt conditions and compare paired image outcomes."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def compare(run: Path) -> None:
    """Require complete matching outputs, then export metrics and paired deltas."""
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text())
    parsed = {}
    summary = []
    for variant, suffix in manifest["prompts"].items():
        files = list(run.rglob(f"*samples_pope_{variant}.jsonl"))
        if len(files) != 1:
            raise ValueError(f"Expected one samples file for {variant}, found {len(files)}")
        rows = {}
        counts = defaultdict(Counter)
        for line in files[0].read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            index = int(record["doc_id"])
            if index in rows or not 0 <= index < len(metadata):
                raise ValueError("Invalid or duplicate doc_id")
            doc = metadata[index]
            metric = record["pope_accuracy"]
            if record["input"] != doc["question"].strip() + suffix:
                raise ValueError(f"Prompt mismatch for {variant}, row {index}")
            if str(doc["question_id"]) != str(metric["question_id"]) or doc["answer"] != metric["ground_truth"]:
                raise ValueError("Dataset alignment mismatch")
            pred = metric["prediction"]
            outcome = {("yes", "yes"): "TP", ("yes", "no"): "FN",
                       ("no", "yes"): "FP", ("no", "no"): "TN"}.get((doc["answer"], pred), "unparsed")
            counts[doc["category"]][outcome] += 1
            rows[index] = {**doc, "prediction": pred, "outcome": outcome}
        if set(rows) != set(range(len(metadata))):
            raise ValueError(f"Incomplete output for {variant}")
        parsed[variant] = rows
        for category, c in sorted(counts.items()):
            tp, fn, fp, tn = (c[k] for k in ("TP", "FN", "FP", "TN"))
            values = {"variant": variant, "category": category, "TP": tp, "FN": fn,
                      "FP": fp, "TN": tn, "unparsed": c["unparsed"]}
            for name, numerator, denominator in [("FPR", fp, fp + tn), ("Recall", tp, tp + fn),
                                                  ("Accuracy", tp + tn, tp + fn + fp + tn),
                                                  ("F1", 2 * tp, 2 * tp + fp + fn)]:
                values[name] = numerator / denominator if denominator else ""
            summary.append(values)
    with (run / "summary.csv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    images = sorted({d["image_source"] for d in metadata})
    rng = np.random.default_rng(20261120)
    sampled = rng.integers(0, len(images), size=(10000, len(images)))
    differences = []
    for category in ("random", "popular", "adversarial"):
        for first, second in (("A", "B"), ("B", "C"), ("A", "C")):
            relevant = [i for i, d in enumerate(metadata) if d["category"] == category]
            wrong_to_right = sum(parsed[first][i]["prediction"] != metadata[i]["answer"] and
                                 parsed[second][i]["prediction"] == metadata[i]["answer"] for i in relevant)
            right_to_wrong = sum(parsed[first][i]["prediction"] == metadata[i]["answer"] and
                                 parsed[second][i]["prediction"] != metadata[i]["answer"] for i in relevant)
            result = {"category": category, "comparison": f"{second} minus {first}",
                      "wrong_to_right": wrong_to_right, "right_to_wrong": right_to_wrong}
            # Unparsed responses preclude an all-image fixed-denominator paired interval.
            if any(parsed[v][i]["outcome"] == "unparsed" for v in (first, second) for i in relevant):
                result["interval_status"] = "Skipped due to unparsed responses; inspect denominators"
            else:
                for name, label, event in (("FPR", "no", "FP"), ("Recall", "yes", "TP")):
                    per_image = defaultdict(int)
                    for i in relevant:
                        if metadata[i]["answer"] == label:
                            per_image[metadata[i]["image_source"]] += (
                                int(parsed[second][i]["outcome"] == event) - int(parsed[first][i]["outcome"] == event))
                    values = np.array([per_image[image] for image in images])
                    delta = values.sum() / (3 * len(images)) * 100
                    ci = np.percentile(values[sampled].sum(axis=1) / (3 * len(images)) * 100, [2.5, 97.5])
                    result[f"{name}_delta_pp"] = float(delta)
                    result[f"{name}_95_ci_pp"] = ci.tolist()
            differences.append(result)
    (run / "paired_comparisons.json").write_text(json.dumps(differences, indent=2))
    print(json.dumps(summary, indent=2))
    print("All conditions complete. Saved summary.csv and paired_comparisons.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    compare(parser.parse_args().run)
