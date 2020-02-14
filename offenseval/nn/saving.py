import os
import pickle
import torch
# We need to load all the models so `load_model` works; sorry for the *
from .models import *

def save_model(model, TEXT, output_path):
    base, _ = os.path.splitext(output_path)
    vocab_path = f"{base}.vocab.pkl"

    torch.save(model, output_path)

    with open(vocab_path, "wb") as f:
        pickle.dump(TEXT, f)

    print(f"Model saved to {output_path}")
    print(f"Vocab saved to {vocab_path}")

def load_model(model_path):
    base, _ = os.path.splitext(model_path)
    vocab_path = f"{base}.vocab.pkl"

    with open(vocab_path, "rb") as f:
        TEXT = pickle.load(f)

    model = torch.load(model_path)

    return model, TEXT
