from typing import Optional, Literal
import os
from dataclasses import dataclass
from behaviors import ALL_BEHAVIORS
from utils.tokenize import get_prompt_template

@dataclass
class SteeringSettings:
    behavior: str = "hallucination"
    type: Literal["open_ended", "ab"] = "ab"
    system_prompt: Optional[Literal["pos", "neg"]] = None
    override_vector: Optional[int] = None
    override_vector_model: Optional[str] = None
    use_base_model: bool = True
    model_size: str = "8b"
    override_model_weights_path: Optional[str] = None
    prompt_template: str = "qa"

    def __post_init__(self):
        assert self.behavior in ALL_BEHAVIORS, f"Invalid behavior {self.behavior}"
        get_prompt_template(self.prompt_template)
        
    def make_result_save_suffix(
        self,
        layer: Optional[int] = None,
        multiplier: Optional[int] = None,
    ):
        elements = {
            "layer": layer,
            "multiplier": multiplier,
            "behavior": self.behavior,
            "type": self.type,
            "system_prompt": self.system_prompt,
            "override_vector": self.override_vector,
            "override_vector_model": self.override_vector_model,
            "model_size": self.model_size,
            "override_model_weights_path": self.override_model_weights_path,
            "prompt_template": self.prompt_template,
        }
        return "_".join([f"{k}={str(v).replace('/', '-')}" for k, v in elements.items() if v is not None])

    def filter_result_files_by_suffix(
        self,
        directory: str,
        layer: Optional[int] = None,
        multiplier: Optional[int] = None,
    ):
        elements = {
            "layer": str(layer)+"_",
            "multiplier": str(float(multiplier))+"_",
            "behavior": self.behavior,
            "type": self.type,
            "system_prompt": self.system_prompt,
            "override_vector": self.override_vector,
            "override_vector_model": self.override_vector_model,
            "model_size": self.model_size,
            "override_model_weights_path": self.override_model_weights_path,
            "prompt_template": self.prompt_template,
        }

        filtered_elements = {k: v for k, v in elements.items() if v is not None}
        remove_elements = {k for k, v in elements.items() if v is None}

        matching_files = []

        for filename in os.listdir(directory):
            if all(f"{k}={str(v).replace('/', '-')}" in filename for k, v in filtered_elements.items()):
                # ensure remove_elements are *not* present
                if all(f"{k}=" not in filename for k in remove_elements):
                    matching_files.append(filename)

        return [os.path.join(directory, f) for f in matching_files]
    
    def get_formatted_model_name(self):
        return "Llama 3.1 8B"
        
