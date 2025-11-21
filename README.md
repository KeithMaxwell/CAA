# Steering Llama 3.1 8B with Contrastive Activation Addition

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then create a `.env` file with the following variables (see `.env.example`):

```
HF_TOKEN=huggingface_token_with_access_to_Meta-Llama-3.1-8B
OPEN_AI_KEY=openai_api_key_with_access_to_gpt4
```

## Dataset

This repository now focuses solely on hallucination reduction. All raw and processed data live under `/datasets/hallucination` and were generated using GPT-4 by Wuschel Schulz [(source)](https://github.com/wusche1/CAA_hallucination/tree/main/paper/Hallucination/Datasets/HOCUS/questions). The generate/test splits are created via `process_raw_datasets.py`, with 50 contrast pairs reserved for evaluation and the remainder used to form steering vectors.

## Evaluation

We evaluate hallucination mitigation with two formats:

- **A/B questions**: contrast pairs with explicit correct/incorrect answers to quantify probability mass on grounded responses.
- **Open-ended prompts**: the same held-out questions without answer options, scored downstream with `scoring.py` to measure hallucination severity.

## Available commands

```bash
# Generate steering vectors for hallucination
python generate_vectors.py --layers $(seq 0 31) --save_activations --model_size "8b" --behaviors hallucination

# Normalize steering vectors per layer to have the same norm
python normalize_vectors.py

# Evaluate hallucination on A/B or open-ended prompts while using CAA
python prompting_with_steering.py --behaviors hallucination --layers $(seq 0 31) --multipliers -1 0 1 --type ab --model_size "8b"
python prompting_with_steering.py --behaviors hallucination --layers 13 --multipliers -2 -1.5 -1 -0.5 0 0.5 1 1.5 2 --type ab --model_size "8b" --system_prompt pos

# Plot PCA of constrastive activations
python plot_activations.py --behaviors hallucination --layers $(seq 0 31) --model_size "8b"

# Plot results of CAA steering effect
python plot_results.py --layers $(seq 0 31) --multipliers 1 --type ab
python plot_results.py --layers $(seq 0 31) --multipliers -1 0 1 --behaviors hallucination --type ab

# Finetune a llama on the hallucination dataset using supervised finetuning on the A/B tokens
python finetune_llama.py --behavior hallucination --direction pos

# Plot similarites of steering vectors
python analyze_vectors.py

# Use GPT-4 to score open-ended responses
python scoring.py
```

All prompts now use a simple `Question: ...\nAnswer:` template without any additional system or fine-tuning instructions to match the base Meta-Llama-3.1-8B model.

## Running tests

I have added a few unit tests for some of the utility functions. To run them, simply run:

```bash
pytest
```

TODO: add more unit tests

## Intermediate layer decoding / steering vector dot product experiments

See `/activation_steering_interp.ipynb`

## Vectors for use

Hallucination steering vectors live in `/vectors/hallucination`. `normalize_vectors.py` rescales each layer to unit norm before running `prompting_with_steering.py` so multipliers are comparable across layers.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
