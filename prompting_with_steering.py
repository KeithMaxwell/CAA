"""
Use CAA to steer the model

Usage:
python prompting_with_steering.py --behaviors hallucination --layers 10 --multipliers 0.1 0.5 1 2 5 10 --type ab --model_size 8b
"""

import json
from llama_wrapper import LlamaWrapper
import os
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
from utils.helpers import get_a_b_probs
from utils.tokenize import E_INST, get_prompt_template
from steering_settings import SteeringSettings
from behaviors import (
    get_open_ended_test_data,
    get_steering_vector,
    get_system_prompt,
    get_ab_test_data,
    HALLUCINATION,
    get_results_dir,
)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

def process_item_ab(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
    prompt_template,
) -> Dict[str, str]:
    question: str = prompt_template.get_input(item)
    answer_matching_behavior = prompt_template.get_positive(item)
    answer_not_matching_behavior = prompt_template.get_negative(item)
    model_output = model.get_logits_from_text(
        user_input=question, model_output="(", system_prompt=system_prompt
    )
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        prompt_template.input_key: question,
        prompt_template.positive_key: answer_matching_behavior,
        prompt_template.negative_key: answer_not_matching_behavior,
        "a_prob": a_prob,
        "b_prob": b_prob,
    }

def process_item_open_ended(
    item: Dict[str, str],
    model: LlamaWrapper,
    system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
    prompt_template,
) -> Dict[str, str]:
    question = prompt_template.get_open_ended_input(item)
    model_output = model.generate_text(
        user_input=question, system_prompt=system_prompt, max_new_tokens=100
    )
    return {
        prompt_template.input_key: question,
        "model_output": model_output.split(E_INST)[-1].strip(),
        "raw_model_output": model_output,
    }


def test_steering(
    layers: List[int], multipliers: List[int], settings: SteeringSettings, overwrite=False
):
    """
    layers: List of layers to test steering on.
    multipliers: List of multipliers to test steering with.
    settings: SteeringSettings object.
    """
    save_results_dir = get_results_dir(settings.behavior)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    process_methods = {
        "ab": process_item_ab,
        "open_ended": process_item_open_ended,
    }
    test_datasets = {
        "ab": get_ab_test_data(settings.behavior),
        "open_ended": get_open_ended_test_data(settings.behavior),
    }
    prompt_template = get_prompt_template(settings.prompt_template)
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN,
        size=settings.model_size,
        override_model_weights_path=settings.override_model_weights_path,
        prompt_template=prompt_template,
    )
    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")
    model.set_save_internal_decodings(False)
    test_data = test_datasets[settings.type]
    for layer in layers:
        name_path = model.model_name_path
        if settings.override_vector_model is not None:
            name_path = settings.override_vector_model
        if settings.override_vector is not None:
            vector = get_steering_vector(settings.behavior, settings.override_vector, name_path, normalized=True)
        else:
            vector = get_steering_vector(settings.behavior, layer, name_path, normalized=True)
        vector = vector.to(model.device).to(next(model.model.parameters()).dtype)
        for multiplier in multipliers:
            result_save_suffix = settings.make_result_save_suffix(
                layer=layer, multiplier=multiplier
            )
            save_filename = os.path.join(
                save_results_dir,
                f"results_{result_save_suffix}.json",
            )
            if os.path.exists(save_filename) and not overwrite:
                print("Found existing", save_filename, "- skipping")
                continue
            results = []
            for item in tqdm(test_data, desc=f"Layer {layer}, multiplier {multiplier}"):
                model.reset_all()
                model.set_add_activations(
                    layer, multiplier * vector
                )
                result = process_methods[settings.type](
                    item=item,
                    model=model,
                    system_prompt=get_system_prompt(settings.behavior, settings.system_prompt),
                    a_token_id=a_token_id,
                    b_token_id=b_token_id,
                    prompt_template=prompt_template,
                )
                results.append(result)
            with open(
                save_filename,
                "w",
            ) as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument(
        "--behaviors",
        type=str,
        nargs="+",
        default=[HALLUCINATION]
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["ab", "open_ended"],
    )
    parser.add_argument("--system_prompt", type=str, default=None, choices=["pos", "neg"], required=False)
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument("--model_size", type=str, choices=["8b"], default="8b")
    parser.add_argument("--override_model_weights_path", type=str, default=None)
    parser.add_argument("--prompt_template", type=str, choices=["qa", "summarization"], default="qa")
    parser.add_argument("--overwrite", action="store_true", default=False)
    
    args = parser.parse_args()

    steering_settings = SteeringSettings()
    steering_settings.type = args.type
    steering_settings.system_prompt = args.system_prompt
    steering_settings.override_vector = args.override_vector
    steering_settings.override_vector_model = args.override_vector_model
    steering_settings.model_size = args.model_size
    steering_settings.override_model_weights_path = args.override_model_weights_path
    steering_settings.prompt_template = args.prompt_template

    for behavior in args.behaviors:
        steering_settings.behavior = behavior
        test_steering(
            layers=args.layers,
            multipliers=args.multipliers,
            settings=steering_settings,
            overwrite=args.overwrite,
        )
