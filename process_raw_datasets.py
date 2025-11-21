import os
import json
import random
import behaviors
import argparse
from utils.tokenize import get_prompt_template

N_TEST = 50
MAX_N_GENERATE = 1000

def make_generate_test_datasets(behavior, prompt_template):
    read_path = behaviors.get_raw_data_path(behavior)
    write_generate_path = behaviors.get_ab_data_path(behavior)
    write_test_path = behaviors.get_ab_data_path(behavior, test=True)
    write_open_ended_test_path = behaviors.get_open_ended_data_path(behavior)
    # make subdirectories if they don't exist
    os.makedirs(os.path.dirname(write_generate_path), exist_ok=True)
    os.makedirs(os.path.dirname(write_test_path), exist_ok=True)
    with open(read_path, "r") as f:
        data = json.load(f)
    cleaned_data = []
    for item in data:
        question = prompt_template.get_input(item)
        answer_matching_behavior = prompt_template.get_positive(item)
        answer_not_matching_behavior = prompt_template.get_negative(item)
        cleaned_data.append(
            {
                prompt_template.input_key: question.replace(prompt_template.input_prefix, "").replace(prompt_template.output_prefix, "").strip(),
                prompt_template.positive_key: answer_matching_behavior.strip(),
                prompt_template.negative_key: answer_not_matching_behavior.strip(),
            }
        )
    random.shuffle(cleaned_data)
    n_generate = min(MAX_N_GENERATE, len(cleaned_data) - N_TEST)
    generate_data = cleaned_data[:n_generate]
    test_data = cleaned_data[-N_TEST:]
    with open(write_generate_path, "w") as f:
        json.dump(generate_data, f, indent=4)
    with open(write_test_path, "w") as f:
        json.dump(test_data, f, indent=4)
    open_ended_test = []
    for item in test_data:
        open_ended_test.append(
            {
                prompt_template.input_key: prompt_template.get_open_ended_input(item),
            }
        )
    with open(write_open_ended_test_path, "w") as f:
        json.dump(open_ended_test, f, indent=4)
    print(f"{behavior}: n_generate: {n_generate} | n_test: {N_TEST}")

def main(args):
    prompt_template = get_prompt_template(args.prompt_template)
    for behavior in args.behaviors:
        make_generate_test_datasets(behavior, prompt_template)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--behaviors", nargs="+", default=behaviors.ALL_BEHAVIORS)
    parser.add_argument("--prompt_template", type=str, choices=["qa", "summarization"], default="qa")
    args = parser.parse_args()
    main(args)
