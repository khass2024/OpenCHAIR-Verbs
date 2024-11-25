import pandas as pd
import sqlite3
import numpy as np
from tqdm.auto import tqdm
import spacy
from functools import lru_cache
from collections import Counter, OrderedDict
import ast
import matplotlib.pyplot as plt
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
    LlamaTokenizer,
    LlamaForCausalLM
)
import torch
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# Hugging face log-in
# https://huggingface.co/meta-llama/Llama-2-7b-hf
hf_token = os.getenv("HF_TOKEN")
login(hf_token) # your_huggingface_token


# Step 1: Load SpaCy for adjective extraction
# python -m spacy download en_core_web_trf
nlp = spacy.load("en_core_web_trf")

@lru_cache(maxsize=None)
def get_adj(caption):
    """Extract adjectives from a caption."""
    doc = nlp(caption.lower())
    return {token.lemma_ for token in doc if token.pos_ == 'ADJ'}

# Step 2: Initialize Llama-7B
def initialize_llama(model_name, cache_dir):
    """Initialize the Llama-7B model and tokenizer."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cache_dir
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        temperature=0.7,
        repetition_penalty=1.2,
        max_new_tokens=30,
        do_sample=True
    )
    return pipe

# Step 3: Expand captions using Llama-7B
def generate_captions(pipe, seed_data, target_size, batch_size=10):
    """Generate new captions using Llama-7B."""
    seed_captions = list(seed_data['Caption'].drop_duplicates())
    new_data = []

    pbar = tqdm(total=target_size, desc="Generating Captions")

    while len(new_data) < target_size:
        prompts = np.random.choice(seed_captions, size=min(batch_size, len(seed_captions)))

        for prompt in prompts:
            refined_prompt = (
                f"Write a one-sentence descriptive caption for an image. "
                f"The caption should be concise and use up to three adjectives. Description: {prompt}"
            )
            outputs = pipe(refined_prompt, num_return_sequences=1)
            for output in outputs:
                generated_caption = output["generated_text"].strip()
                # Ensure unique captions and limit attributes to 3
                if generated_caption not in [row['Caption'] for row in new_data]:
                    visual_attrs = list(
                        seed_data[seed_data['Caption'] == prompt]['Visual Attributes'].iloc[0]
                    )[:3]
                    new_data.append({"Caption": generated_caption, "Visual Attributes": visual_attrs})
                    pbar.update(1)
                    if len(new_data) >= target_size:
                        break
    pbar.close()
    return pd.DataFrame(new_data)

# Step 4: Filter captions based on adjectives and visual checks
def adj_visual_check(pipe, sentence, adjs):
    """Run visual adjective checks on a caption."""
    visual_tags = []
    for adj in adjs:
        prompt = f"Does the sentence '{sentence}' use '{adj}' as a visual adjective for an image? Answer with yes or no."
        # prompt = f"Is '{adj}' in the sentence '{sentence}' a visual adjective? Answer with yes or no."
        result = pipe(prompt, max_new_tokens=10, temperature=0.8, num_return_sequences=1, do_sample=True)
        output_text = result[0]['generated_text'].strip().lower()
        if "yes" in output_text:
            visual_tags.append('yes')
        elif "no" in output_text:
            visual_tags.append('no')
        else:
            visual_tags.append('unsure')
    return visual_tags

# def filter_and_validate_captions(data, pipe):
#     """Filter captions to retain those with diverse and valid adjectives."""
#     valid_data = []
#     adj_counts = Counter()

#     for _, row in tqdm(data.iterrows(), total=len(data), desc="Filtering Captions"):
#         caption = row['Caption']
#         visual_attrs = row['Visual Attributes']
#         adjs = get_adj(caption)
#         if 0 < len(adjs) <= 3:  # Limit number of adjectives
#             visual_tags = adj_visual_check(pipe, caption, adjs)
#             if "yes" in visual_tags:
#                 valid_data.append({"Caption": caption, "Visual Attributes": visual_attrs})
#                 adj_counts.update(adjs)
    
#     return pd.DataFrame(valid_data), adj_counts
def filter_and_validate_captions(data, pipe):
    """Filter captions to retain those with diverse and valid adjectives."""
    valid_data = []
    adj_counts = Counter()

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Filtering Captions"):
        caption = row['Caption']
        visual_attrs = row['Visual Attributes']
        adjs = get_adj(caption)

        if 0 < len(adjs) <= 3:  # Limit number of adjectives
            visual_tags = adj_visual_check(pipe, caption, adjs)
            if "yes" in visual_tags:
                valid_data.append({"Caption": caption, "Visual Attributes": visual_attrs})
                adj_counts.update(adjs)
            else:
                print(f"Caption rejected: {caption}, Adjectives: {adjs}, Tags: {visual_tags}")
        else:
            print(f"Caption skipped due to adjective count: {caption}, Adjectives: {adjs}")

    return pd.DataFrame(valid_data), adj_counts

# Step 5: Ensure diversity by prioritizing rare adjectives
def ensure_diversity(data, adjective_distribution, threshold=0.85):
    """Ensure diversity by prioritizing captions with rare adjectives."""
    total_adjectives = sum(adjective_distribution.values())
    rare_adjectives = {
        adj for adj, count in adjective_distribution.items()
        if count / total_adjectives < threshold
    }
    diverse_data = [
        row for _, row in data.iterrows()
        if any(adj in rare_adjectives for adj in get_adj(row['Caption']))
    ]
    return pd.DataFrame(diverse_data)

# Step 6: Main script
if __name__ == "__main__":
    # Paths and parameters
    seed_data_path = "/home/jinnylee/DL/adj_seed.csv"
    cache_dir = "/home/jinnylee/DL/cache_dir"
    model_name = "meta-llama/Llama-2-7b-hf"
    target_size = 20
    output_path = "/home/jinnylee/DL/final_captions.csv"


    # Load seed data
    seed_data = pd.read_csv(seed_data_path)

    # Initialize Llama-7B
    pipe = initialize_llama(model_name, cache_dir)

    # Generate new captions
    expanded_data = generate_captions(pipe, seed_data, target_size)
    expanded_data.to_csv("/home/jinnylee/DL/expanded_captions.csv", index=False)

    # Filter and validate captions
    filtered_data, adjective_distribution = filter_and_validate_captions(expanded_data, pipe)
    filtered_data.to_csv("/home/jinnylee/DL/filtered_captions.csv", index=False)

    # Ensure diversity
    final_data = ensure_diversity(filtered_data, adjective_distribution)

    # Save final dataset
    final_data.to_csv(output_path, index=False)

    print(f"Final dataset saved to {output_path} with {len(final_data)} entries.")