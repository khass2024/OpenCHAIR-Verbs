import os
import pandas as pd
from datasets import Dataset, DatasetDict
from PIL import Image
from huggingface_hub import login


# Step 1: Log in to Hugging Face
def huggingface_login(token):
    login(token=token)  # Replace with your actual token


# Step 2: Load the CSV file
def load_csv(csv_path):
    return pd.read_csv(csv_path)


# Step 3: Load images based on the DataFrame index (loc)
def load_image(example, folder_name, idx):
    image_path = os.path.join(folder_name, 'images', f'{idx:06d}.jpg')  # Adjust file extension if necessary
    image = Image.open(image_path)
    return {"image": image}


# Step 4: Create Hugging Face Dataset with only the image and caption (no extra columns)
def create_dataset(df, folder_name):
    df = df[['caption']]  # Select only the caption column
    dataset = Dataset.from_pandas(df)

    # Map function with the index from df, keeping only the image and caption
    dataset = dataset.map(lambda example, idx: {**load_image(example, folder_name, idx+1), "text": example["caption"]},
                          with_indices=True)

    # Remove the 'cap' column (if still present)
    dataset = dataset.remove_columns(['caption'])

    return dataset


# Step 5: Save dataset to Hugging Face Hub
def push_to_hub(dataset, username, dataset_name):
    dataset_dict = DatasetDict({"test": dataset})
    dataset_dict.push_to_hub(f"{username}/{dataset_name}")

# Main script
if __name__ == "__main__":
    # Step 1: Log in to Hugging Face
    token = 'your_hf_token'
    huggingface_login(token)

    # Step 2: Load CSV file containing captions
    folder_name = 'verb_captions'
    csv_file = os.path.join(folder_name, 'data', 'filtered_verbs_llm.csv')
    df = load_csv(csv_file)

    # Step 3: Create Hugging Face Dataset
    dataset = create_dataset(df, folder_name)

    # Step 4: Push dataset to Hugging Face Hub
    username = "User"  # Replace with your Hugging Face username
    dataset_name = "dataset_name"  # Replace with the desired dataset name
    push_to_hub(dataset, username, dataset_name)

    print(f"Dataset pushed to Hugging Face Hub: {username}/{dataset_name}")
