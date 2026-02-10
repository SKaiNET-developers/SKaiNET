"""
Generate reference data for BERT numerical accuracy tests.

Loads MongoDB/mdbr-leaf-ir via sentence-transformers, runs inference on test
inputs, and saves token IDs, hidden states, final embeddings, and cosine
similarity to reference_data.json.

Usage:
    cd bert-research && uv run python generate_reference.py
"""

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "MongoDB/mdbr-leaf-ir"
TEST_INPUTS = ["hello world", "retrieval augmented generation"]
OUTPUT_FILE = "reference_data.json"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print(f"Loading model: {MODEL_NAME}")

    # Load HuggingFace tokenizer and base model for hidden states
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    base_model.eval()

    # Load sentence-transformers model for full pipeline (pooling + projection + normalize)
    st_model = SentenceTransformer(MODEL_NAME)

    reference = {"model": MODEL_NAME, "inputs": []}

    for text in TEST_INPUTS:
        print(f"\nProcessing: \"{text}\"")

        # Tokenize
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"][0].tolist()
        attention_mask = encoded["attention_mask"][0].tolist()
        token_type_ids = encoded["token_type_ids"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        print(f"  Token IDs: {input_ids}")
        print(f"  Tokens: {tokens}")

        # Run base model for hidden states
        with torch.no_grad():
            outputs = base_model(**encoded)
            hidden_states = outputs.last_hidden_state[0]  # [seqLen, hiddenSize]

        hidden_np = hidden_states.numpy()
        print(f"  Hidden states shape: {hidden_np.shape}")
        print(f"  Hidden[0][:8]: {hidden_np[0][:8].tolist()}")

        # Run sentence-transformers full pipeline for final embedding
        embedding = st_model.encode(text, normalize_embeddings=True)
        print(f"  Final embedding shape: {embedding.shape}")
        print(f"  Embedding[:8]: {embedding[:8].tolist()}")

        entry = {
            "text": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "tokens": tokens,
            "hidden_states": hidden_np.tolist(),
            "embedding": embedding.tolist(),
        }
        reference["inputs"].append(entry)

    # Compute cosine similarity between the two embeddings
    emb_a = np.array(reference["inputs"][0]["embedding"])
    emb_b = np.array(reference["inputs"][1]["embedding"])
    sim = cosine_similarity(emb_a, emb_b)
    reference["cosine_similarity"] = sim
    print(f"\nCosine similarity between inputs: {sim:.6f}")

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"\nSaved reference data to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
