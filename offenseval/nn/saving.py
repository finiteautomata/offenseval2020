import os
import pickle
import torch

def save_model(model, TEXT, output_path):
    base, _ = os.path.splitext(output_path)
    vocab_path = f"{base}.vocab.pkl"
    num_score_path = f"{base}.num_score.pkl"

    torch.save(model, output_path)

    with open(vocab_path, "wb") as f:
        pickle.dump(TEXT, f)

    print(f"Model saved to {output_path}")
    print(f"Vocab saved to {vocab_path}")
