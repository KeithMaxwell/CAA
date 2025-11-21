from behaviors import ALL_BEHAVIORS, get_vector_path
from utils.helpers import get_model_path
import torch as t
import os

def normalize_vectors(model_size: str, n_layers: int):
    # make normalized_vectors directory
    normalized_vectors_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalized_vectors")
    if not os.path.exists(normalized_vectors_dir):
        os.makedirs(normalized_vectors_dir)
    for layer in range(n_layers):
        print(layer)
        for behavior in ALL_BEHAVIORS:
            vec_path = get_vector_path(behavior, layer, get_model_path(model_size))
            vec = t.load(vec_path)
            new_path = vec_path.replace("vectors", "normalized_vectors")
            if not os.path.exists(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path))
            t.save(vec / vec.norm(), new_path)
    
    
if __name__ == "__main__":
    normalize_vectors("8b", 32)