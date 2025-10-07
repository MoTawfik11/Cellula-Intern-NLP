# dataset_prep.py
from datasets import load_dataset
import json
from tqdm import tqdm

def prepare_humaneval(output_file="data/humaneval.jsonl"):
    ds = load_dataset("openai/openai_humaneval", split="test")
    with open(output_file, "w", encoding="utf-8") as f:
        for i, ex in enumerate(tqdm(ds)):
            task = {
                "task_id": ex.get("task_id", f"humaneval_{i}"),
                "prompt": ex["prompt"],
                "canonical_solution": ex.get("canonical_solution", "")
            }
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    prepare_humaneval()
