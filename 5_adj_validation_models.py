from datasets import load_dataset, DatasetDict
from transformers import pipeline
import spacy
from collections import Counter
from tqdm import tqdm
import pandas as pd
import torch
from dotenv import load_dotenv
import os

from huggingface_hub import login

load_dotenv()


def huggingface_login(token):
    login(token=token)  # Replace with your actual token

# Step 1: Load Dataset from Hugging Face
def load_huggingface_dataset(dataset_path):
    """Load the dataset from Hugging Face Hub."""
    return load_dataset(dataset_path)

# Step 2: Initialize Captioning Models
def initialize_captioning_models():
    """Initialize multiple state-of-the-art captioning models."""
    device = 0 if torch.cuda.is_available() else -1
    models = {
        "BLIP": pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=device),
        "GIT": pipeline("image-to-text", model="microsoft/git-large", device=device),
        # "CLIPCap": pipeline("image-to-text", model="openai/clipcap", device=device)
        "VIT-GPT2": pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", device=device),
        "BLIP-Large": pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=device)
    }
    return models

# Step 3: Generate Captions for Each Image
# def generate_captions_for_images(dataset, models):
#     """Generate captions using multiple models."""
#     captions = []
#     for idx, image in tqdm(enumerate(dataset["image"]), total=len(dataset["image"]), desc="Generating Captions"):
#         try:
#             # Use index as the unique identifier
#             image_captions = {"image_id": f"image_{idx}"}
#             for model_name, model in models.items():
#                 image_captions[model_name] = model(image, max_new_tokens=30)[0]["generated_text"]
#             captions.append(image_captions)
#         except Exception as e:
#             print(f"Error generating caption for image {idx}: {e}")
#     return pd.DataFrame(captions)
def generate_captions_for_images(dataset, models):
    """Generate captions using multiple models."""
    captions = []
    for idx, image in tqdm(enumerate(dataset["image"]), total=len(dataset["image"]), desc="Generating Captions"):
        try:
            # Add the original caption to the output
            original_caption = dataset[idx]["text"]  # Ensure "text" exists in the dataset
            image_captions = {"image_id": f"image_{idx}", "original_caption": original_caption}
            for model_name, model in models.items():
                image_captions[model_name] = model(image, max_new_tokens=30)[0]["generated_text"]
            captions.append(image_captions)
        except Exception as e:
            print(f"Error generating caption for image {idx}: {e}")
    return pd.DataFrame(captions)

# Step 4: Extract Adjectives Using spaCy
def extract_adjectives(caption):
    """Extract adjectives from a caption."""
    doc = nlp(caption)
    return [token.text for token in doc if token.pos_ == "ADJ"]

# def extract_all_adjectives(captions):
#     """Extract adjectives from all captions."""
#     adjectives = {}
#     for _, row in tqdm(captions.iterrows(), total=len(captions), desc="Extracting Adjectives"):
#         # Use the image_id as the key
#         adjectives[row["image_id"]] = {
#             model: extract_adjectives(row[model]) for model in row.keys() if model != "image_id"
#         }
#     return adjectives

def extract_all_adjectives(captions):
    """Extract adjectives from all captions."""
    adjective_records = []  # Collect data for saving as a CSV
    adjectives = {}
    for _, row in tqdm(captions.iterrows(), total=len(captions), desc="Extracting Adjectives"):
        image_id = row["image_id"]
        adj_dict = {}
        for model in row.keys():
            if model not in ["image_id", "original_caption"]:
                adj_dict[model] = extract_adjectives(row[model])
        adjectives[image_id] = adj_dict
        adjective_records.append({"image_id": image_id, **adj_dict, "original_caption": row["original_caption"]})
    # Save all captions and extracted adjectives to CSV for analysis
    pd.DataFrame(adjective_records).to_csv("adjectives_per_model.csv", index=False)
    return adjectives


# Step 5: Voting Mechanism for Adjectives
# def vote_on_adjectives(adjective_dict, threshold=2):
#     """Filter adjectives based on voting threshold."""
#     final_adjectives = {}
#     for image, adj_dict in adjective_dict.items():
#         all_adjectives = [adj for model_adjs in adj_dict.values() for adj in model_adjs]
#         counts = Counter(all_adjectives)
#         final_adjectives[image] = [adj for adj, count in counts.items() if count >= threshold]
#     return final_adjectives

def vote_on_adjectives(adjective_dict, threshold=2):
    """Filter adjectives based on voting threshold."""
    final_adjectives = []
    for image, adj_dict in adjective_dict.items():
        all_adjectives = [adj for model_adjs in adj_dict.values() for adj in model_adjs]
        counts = Counter(all_adjectives)
        final_adjectives.append({
            "image_id": image,
            "voted_adjectives": [adj for adj, count in counts.items() if count >= threshold]
        })
    # Save the voted adjectives to CSV for review
    pd.DataFrame(final_adjectives).to_csv("voted_adjectives.csv", index=False)
    return {item["image_id"]: item["voted_adjectives"] for item in final_adjectives}



# Step 6: Filter and Save Dataset
# def filter_and_save_dataset(dataset, voted_adjectives, output_path):
#     """Filter dataset and save results."""
#     filtered_data = []
#     for idx in tqdm(range(len(dataset["image"])), desc="Filtering Dataset"):
#         image_id = f"image_{idx}"
#         if voted_adjectives.get(image_id):
#             filtered_data.append({
#                 "image_id": image_id,
#                 "voted_adjectives": voted_adjectives[image_id]
#             })
#     pd.DataFrame(filtered_data).to_csv(output_path, index=False)
#     print(f"Filtered dataset saved to {output_path}")

def save_combined_results(captions_df, adjective_dict, voted_adjectives, output_path):
    """Combine all results and save them in a single CSV file."""
    combined_data = []
    for _, row in tqdm(captions_df.iterrows(), total=len(captions_df), desc="Saving Results"):
        image_id = row["image_id"]
        combined_data.append({
            "image_id": image_id,
            "original_caption": row["original_caption"],
            "BLIP_caption": row["BLIP"],
            "GIT_caption": row["GIT"],
            "VIT-GPT2_caption": row["VIT-GPT2"],
            "BLIP-Large_caption": row["BLIP-Large"],
            "voted_adjectives": voted_adjectives.get(image_id, []),
        })
    pd.DataFrame(combined_data).to_csv(output_path, index=False)
    print(f"Combined results saved to {output_path}")


# Main Script
if __name__ == "__main__":
    hf_token = os.getenv("HF_TOKEN")
    huggingface_login(hf_token)

    # Step 1: Load dataset from Hugging Face
    dataset_path = "wlsdml357/adj_image_caption_pair"  # Replace with your dataset path on Hugging Face
    dataset = load_huggingface_dataset(dataset_path)["test"]

    # # **Process Only the First 10 Images**
    # dataset = load_huggingface_dataset(dataset_path)["test"].select(range(10))
    
    # Step 2: Initialize spaCy and captioning models
    nlp = spacy.load("en_core_web_sm")
    models = initialize_captioning_models()

    # Step 3: Generate captions for images
    captions = generate_captions_for_images(dataset, models)

    # Step 4: Extract adjectives
    adjective_dict = extract_all_adjectives(captions)

    # Step 5: Conduct voting on adjectives
    threshold = 2  # Set your desired voting threshold
    voted_adjectives = vote_on_adjectives(adjective_dict, threshold=threshold)

    # Step 6: Filter dataset and save results
    output_path = "filtered_image_caption_pair.csv"
    save_combined_results(captions, adjective_dict, voted_adjectives, output_path)
    # filter_and_save_dataset(dataset, voted_adjectives, output_path)