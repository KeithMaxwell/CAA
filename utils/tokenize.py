from dataclasses import dataclass
from typing import Callable, List, Optional
from transformers import PreTrainedTokenizer


@dataclass
class PromptTemplate:
    """Simple wrapper to keep prompt formatting consistent across tasks."""

    name: str
    input_prefix: str
    output_prefix: str
    input_key: str
    positive_key: str
    negative_key: str
    open_ended_input_transform: Optional[Callable[[str], str]] = None

    def format(self, user_input: str, model_output: Optional[str] = None) -> str:
        prompt = f"{self.input_prefix} {user_input.strip()}{self.output_prefix}"
        if model_output is not None:
            prompt += f" {model_output.strip()}"
        return prompt

    def get_input(self, item: dict) -> str:
        return item[self.input_key]

    def get_positive(self, item: dict) -> str:
        value = item[self.positive_key]
        if isinstance(value, list):
            value = value[0]
        return value

    def get_negative(self, item: dict) -> str:
        value = item[self.negative_key]
        if isinstance(value, list):
            value = value[0]
        return value

    def get_open_ended_input(self, item: dict) -> str:
        base = self.get_input(item)
        if self.open_ended_input_transform is None:
            return base.strip()
        return self.open_ended_input_transform(base)


PROMPT_TEMPLATES = {
    "qa": PromptTemplate(
        name="qa",
        input_prefix="Question:",
        output_prefix="\nAnswer:",
        input_key="question",
        positive_key="answer_matching_behavior",
        negative_key="answer_not_matching_behavior",
        open_ended_input_transform=lambda text: text.split("\n")[0].strip(),
    ),
    "summarization": PromptTemplate(
        name="summarization",
        input_prefix="Article:",
        output_prefix="\nSummary:",
        input_key="article",
        positive_key="summary_matching_behavior",
        negative_key="summary_not_matching_behavior",
    ),
}


def get_prompt_template(name: str = "qa") -> PromptTemplate:
    try:
        return PROMPT_TEMPLATES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt template '{name}'") from exc


def tokenize_llama_base(
    tokenizer: PreTrainedTokenizer,
    prompt_template: PromptTemplate,
    user_input: str,
    model_output: Optional[str] = None,
) -> List[int]:
    input_content = prompt_template.format(user_input=user_input, model_output=model_output)
    return tokenizer.encode(input_content)
