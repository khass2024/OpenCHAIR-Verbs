import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import random
from dotenv import load_dotenv
import os

load_dotenv()

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hugging face log-in
from huggingface_hub import login
# https://huggingface.co/meta-llama/Llama-2-7b-hf
hf_token = os.getenv("HF_TOKEN")
login(hf_token) # your_huggingface_token

# Load the tokenizer and model
model_name = "meta-llama/Llama-2-7b-hf"  
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model in half precision to save memory, if using CUDA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

# Create a text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

# Function to generate captions
def generate_captions(num_captions, prompt_template):
    captions = set()
    for _ in range(num_captions):
        # Vary the prompt slightly by inserting random words or phrases
        prompt = prompt_template.format(random_word=random.choice([
            "beautiful", "ancient", "modern", "mysterious", "vibrant", "serene", "tall", "green", "young"
        ]))

        # Generate the caption
        output = generator(
            prompt,
            max_length=40,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )

        # Extract the generated text and clean it up
        caption = output[0]["generated_text"].strip()
        captions.add(caption)

    return list(captions)

# Define the prompt template
prompt_template = (
    "Generate a unique and descriptive image caption that includes vivid adjectives "
    "and varies in subject, timeline, and location. Start your caption with 'A {random_word}' and continue."
    "Ignore whether the described situation is possible or not."
    "Describe an image with 'A {random_word} scene' and vivid details.",
    "Write a caption for an image that showcases a '{random_word} capybara' in nature.",
    "Create a one-sentence caption describing a '{random_word} woman' and continue.",
    "Generate a descriptive caption for an image of 'A {random_word} building' and continue.",
    "Write a caption for 'A {random_word} photo' taken during golden hour."
)

# Generate captions
num_captions = 100  # Adjust the number as needed
captions = generate_captions(num_captions, prompt_template)

# Optional: Analyze and reduce repetitiveness
# For simplicity, we're using a set to automatically remove exact duplicates

# Print the generated captions
for idx, caption in enumerate(captions, 1):
    print(f"{idx}: {caption}")

num_captions = 5
captions = generate_captions(num_captions, prompt_template)






# # Example usage
# if __name__ == "__main__":
#     model_name = "meta-llama/Llama-2-7b-hf"
#     num_seeds = 10  # Total captions to generate
#     batch_size = 2  # Number of captions per batch

#     seed_df = generate_seed_captions_batch(model_name, num_seeds, batch_size)

#     # Save to CSV
#     seed_csv_path = "llama_adj_seed_batch.csv"
#     seed_df.to_csv(seed_csv_path, index=False)
#     print(f"Seed captions saved to {seed_csv_path}")