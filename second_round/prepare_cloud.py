"""Create local POPE task variants while preserving the installed scorer."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PROMPTS = {
    "A": "\nAnswer the question using a single word or phrase.",
    "B": "\nAnswer yes or no.",
    "C": "\nBase your answer on visible evidence in the image. Answer yes or no.",
}
REVISION = "4db1276663dfa5eb8ad16a52d24c31a09e470896"


def prepare(phase: str, project: Path) -> None:
    """Generate task YAMLs, matching row metadata, and the custom split filter."""
    from datasets import load_dataset

    root = Path(__file__).resolve().parent
    split = json.loads((root / "image_split.json").read_text())
    original = project / "lmms-eval/lmms_eval/tasks/pope"
    task_dir = root / "tasks" / phase
    task_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original / "utils.py", task_dir / "phase2_utils.py")
    source = (original / "pope.yaml").read_text()
    if 'task: "pope"' not in source or "token: True" not in source:
        raise ValueError("Installed POPE YAML differs. Inspect it before generating tasks.")
    # Keep the exact installed prompt formatting and scorer functions.
    extra = f'''
import json as _json
from pathlib import Path as _Path
_SELECTED_IMAGES = set(_json.loads((_Path(__file__).parent / "selected_images.json").read_text()))

def phase2_process_docs(dataset):
    return dataset.filter(lambda doc: doc["image_source"] in _SELECTED_IMAGES)
'''
    with (task_dir / "phase2_utils.py").open("a", encoding="utf-8") as destination:
        destination.write(extra)
    (task_dir / "selected_images.json").write_text(json.dumps(split[phase]))
    for variant, prompt in PROMPTS.items():
        task = source.replace('task: "pope"', f'task: "pope_{variant}"')
        task = task.replace("token: True", f'token: True\n  revision: "{REVISION}"')
        task = task.replace("!function utils.", "!function phase2_utils.")
        task = task.replace("test_split: test", "test_split: test\nprocess_docs: !function phase2_utils.phase2_process_docs")
        # The original task applies the default post_prompt to this model.
        old = 'post_prompt: "\\nAnswer the question using a single word or phrase."'
        if old not in task:
            raise ValueError("Installed default prompt changed")
        task = task.replace(old, "post_prompt: " + json.dumps(prompt))
        (task_dir / f"pope_{variant}.yaml").write_text(task, encoding="utf-8")
    dataset = load_dataset("lmms-lab-encoder/POPE", revision=REVISION, split="test")
    rows = [row for row in dataset.remove_columns("image") if row["image_source"] in set(split[phase])]
    if len(rows) != len(split[phase]) * 18:
        raise ValueError("Unexpected number of questions")
    (root / f"metadata_{phase}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    manifest = {"phase": phase, "images": len(split[phase]), "questions_per_variant": len(rows),
                "revision": REVISION, "prompts": PROMPTS,
                "scorer_sha256": hashlib.sha256((original / "utils.py").read_bytes()).hexdigest()}
    (task_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["dev", "test"])
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    prepare(args.phase, args.project)
