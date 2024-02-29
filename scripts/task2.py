import argparse
import torch 
import h5py

import datasets
import numpy as np
import transformers

MODEL_NAME = "facebook/xglm-564M"
DATASET_NAME = "facebook/flores"

# this is the minimal set of languages that you should analyze
# feel free to experiment with additional lanuages available in the flores dataset
LANGUAGES = [
    "eng_Latn",
    "spa_Latn",
    "deu_Latn",
    "arb_Arab",
    "tam_Taml",
    "quy_Latn"
]

########################################################
# Task 1: Obtain Hidden Representations
########################################################

# Function to obtain hidden representations from the model
def obtain_hidden_representations(model, tokenizer, dataset):
    # TODO: Initialize lists to store representations and tokens
    representations = []
    tokens = []
    
    for example in dataset:
        language = example["language"]
        sentences = example["sentences"]
        
        for sentence in sentences:
            # Tokenize sentence
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"]

            # Forward pass through the model
            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state

            # Save representations for each token
            for token_index, token_id in enumerate(input_ids[0]):
                if token_id.item() != tokenizer.pad_token_id:
                    representation = hidden_states[0, token_index].numpy()
                    representations.append(representation)
                    tokens.append(tokenizer.decode(token_id))

    return representations, tokens



########################################################
# Entry point
########################################################

if __name__ == "__main__":
    # TODO: your code goes here
    # Load model and tokenizer
    model = transformers.AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = datasets.load_dataset(DATASET_NAME)
    print(dataset)
    representations, tokens = obtain_hidden_representations(model, tokenizer, dataset)

