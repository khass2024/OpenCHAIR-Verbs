import os
import pandas as pd
from datasets import Dataset, DatasetDict
from PIL import Image
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# Step 1: Log in to Hugging Face
def huggingface_login(token):
    login(token=token)  # Replace with your actual token


# Step 2: Load the CSV file
def load_csv(csv_path):
    return pd.read_csv(csv_path)


# Step 3: Load images based on the DataFrame index (loc)
def load_image(example, folder_name, idx):
    # Adjust indexing to match image filenames
    image_path = os.path.join(folder_name, 'images', f'{idx:06d}.jpg')
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return {"image": image}


# Step 4: Create Hugging Face Dataset with only the image and caption (no extra columns)
def create_dataset(df, folder_name, max_images=None):
    df = df[['Caption']]  # Select only the caption column
    
    # Limit the dataset to match the number of available images
    if max_images:
        df = df.iloc[:max_images]  # Only take rows for which images exist

    dataset = Dataset.from_pandas(df)

    # Map function with the index from df, keeping only the image and caption
    dataset = dataset.map(lambda example, idx: {**load_image(example, folder_name, idx), "text": example["Caption"]},
                          with_indices=True)

    # Remove the 'Caption' column
    dataset = dataset.remove_columns(['Caption'])
    return dataset


# Step 5: Save dataset to Hugging Face Hub
def push_to_hub(dataset, username, dataset_name):
    dataset_dict = DatasetDict({"test": dataset})
    dataset_dict.push_to_hub(f"{username}/{dataset_name}")

# Main script
if __name__ == "__main__":
    # Step 1: Log in to Hugging Face
    hf_token = os.getenv("HF_TOKEN")
    huggingface_login(hf_token)

    # Step 2: Load CSV file containing captions
    folder_name = '/home/jinnylee/DL'
    csv_file = os.path.join(folder_name, 'adj_seed.csv')
    df = load_csv(csv_file)

    # Step 3: Create Hugging Face Dataset
    max_images = 100  # Adjust this to the number of images you currently have
    dataset = create_dataset(df, folder_name, max_images=max_images)

    # Inspect the dataset before uploading
    for i, entry in enumerate(dataset):
        print(f"Entry {i}: {entry}")
        if i >= 5:  # Only show the first 5 for brevity
            break

    # Step 4: Push dataset to Hugging Face Hub
    username = "wlsdml357"  # Replace with your Hugging Face username
    dataset_name = "adj_image_caption_pair"  # Replace with the desired dataset name
    # Uncomment the next line only when you're ready to push
    push_to_hub(dataset, username, dataset_name)

    print(f"Dataset is ready to be pushed to Hugging Face Hub: {username}/{dataset_name}")