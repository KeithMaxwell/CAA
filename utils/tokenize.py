from typing import List
from transformers import PreTrainedTokenizer

QUESTION_PREFIX = "Question:"
ANSWER_PREFIX = "\nAnswer:"

ADD_FROM_POS_BASE = ANSWER_PREFIX


def tokenize_llama_base(
    tokenizer: PreTrainedTokenizer, user_input: str, model_output: str = None
) -> List[int]:
    input_content = f"{QUESTION_PREFIX} {user_input.strip()}{ANSWER_PREFIX}"
    if model_output is not None:
        input_content += f" {model_output.strip()}"
    return tokenizer.encode(input_content)
