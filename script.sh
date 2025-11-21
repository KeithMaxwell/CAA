python generate_vectors.py --layers $(seq 0 31) --save_activations --model_size "8b" --behaviors hallucination

python normalize_vectors.py

python plot_activations.py --layers $(seq 0 31) --model_size "8b"
python analyze_vectors.py

python prompting_with_steering.py --layers $(seq 0 31) --multipliers -1 0 1 --type ab --model_size "8b" --behaviors hallucination
python prompting_with_steering.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "8b" --behaviors hallucination
python prompting_with_steering.py --layers 13 --multipliers -2.0 -1.5 -1 0 1 1.5 2.0 --type open_ended --model_size "8b" --behaviors hallucination

python plot_results.py --layers $(seq 0 31) --multipliers -1 1 --type ab --model_size "8b"
python plot_results.py --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --title "Layer 13 - Llama 3.1 8B" --model_size "8b"
python plot_results.py --layers 13 --multipliers -1.5 -1 0 1 1.5 --type open_ended --title "Layer 13 - Llama 3.1 8B" --model_size "8b"

python scoring.py
