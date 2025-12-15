import json
import random
from typing import List, Dict


def transform_example(example: Dict, rng: random.Random) -> Dict:
    question = example["question"].strip()

    choices = [
        ("matching", example["answer_matching_behavior"].strip()),
        ("not_matching", example["answer_not_matching_behavior"].strip()),
    ]

    rng.shuffle(choices)

    choice_map = {}
    choice_texts = []

    for idx, (label, text) in enumerate(choices):
        option = "(A)" if idx == 0 else "(B)"
        choice_map[label] = option
        choice_texts.append(f" {option} {text}")

    new_question = (
        f"{question}\n\n"
        "Choices:\n"
        + "\n".join(choice_texts)
    )

    return {
        "question": new_question,
        "answer_matching_behavior": choice_map["matching"],
        "answer_not_matching_behavior": choice_map["not_matching"],
    }


def transform_file(input_path: str, output_path: str, seed: int = 42):
    rng = random.Random(seed)

    with open(input_path, "r", encoding="utf-8") as f:
        data: List[Dict] = json.load(f)

    transformed = [transform_example(item, rng) for item in data]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transformed, f, indent=2, ensure_ascii=False)


transform_file(
    input_path="data/input.json",
    output_path="data/output.json",
    seed=42,   # 改这个可以得到不同随机顺序
)

print("✅ 随机化转换完成")

