import pandas as pd
from datasets import load_dataset
from PIL import Image
import keyboard  # For detecting key presses
import os

# Load the Hugging Face dataset, using 'test' split
dataset = load_dataset('Hagarsh/OpenCHAIR_adj_exp_2', split='test')

# Check how many images are in the dataset
total_images = len(dataset)
images_per_person = total_images // 3  # Roughly divide the dataset into 3 parts

# Prompt to ask which person is working (1, 2, or 3)
print("Please enter your person number (1, 2, or 3):")
person_number = int(input().strip())

# Determine the section of the dataset this person will work on
if person_number == 1:
    start_index = 0
    end_index = images_per_person
elif person_number == 2:
    start_index = images_per_person
    end_index = 2 * images_per_person
else:
    start_index = 2 * images_per_person
    end_index = total_images

print(f"Person {person_number} will work on images from {start_index} to {end_index}.")

# Check if a checkpoint exists for this person
checkpoint_file = f'checkpoint_person_{person_number}.txt'

if os.path.exists(checkpoint_file):
    # Ask the user if they want to restart or resume from the last checkpoint
    print(f"Checkpoint found for Person {person_number}. Do you want to resume from the last checkpoint (y/n)?")
    user_choice = input().strip().lower()

    if user_choice == 'y':
        # If the user wants to resume, load the last index from the checkpoint
        with open(checkpoint_file, 'r') as f:
            last_index = int(f.read().strip())
        print(f"Resuming from the last checkpoint: Image {last_index}")
    else:
        # If the user chooses to restart, start from the beginning of the section
        last_index = start_index
        print(f"Restarting from the beginning of your assigned section: Image {start_index}")
else:
    # If no checkpoint exists, start from the beginning
    last_index = start_index
    print("No checkpoint found. Starting from the beginning of your assigned section.")

# Lists to store good and bad data
good_data_csv = f'good_data_pairs_person_{person_number}.csv'
bad_data_csv = f'bad_data_pairs_person_{person_number}.csv'

def load_csv_file(file_path):
    """Helper function to load CSV files if they exist and have content."""
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            return pd.read_csv(file_path).to_dict('records')
        except pd.errors.EmptyDataError:
            print(f"Warning: {file_path} is empty. Starting with an empty list.")
            return []
    else:
        return []

# Load the good and bad data CSVs
good_data = load_csv_file(good_data_csv)
bad_data = load_csv_file(bad_data_csv)

# Initialize the user_response variable to handle cases where the loop might not execute
user_response = None

# Function to display image and caption, and wait for user response
def display_image_with_caption(image_number, caption):
    # Get the image from the Hugging Face dataset
    image = dataset[image_number]['image']

    # Display the image using Pillow
    image.show()
    print(f"Caption: {caption}")

    # Wait for user input (1, 0, or q) using keyboard module
    print("Press 1 for good data, 0 for bad data, or 'q' to quit.")

    # Keep waiting until valid input is detected
    while True:
        if keyboard.is_pressed('1'):
            print("You pressed 1: Good data")
            return '1'
        elif keyboard.is_pressed('0'):
            print("You pressed 0: Bad data")
            return '0'
        elif keyboard.is_pressed('q'):
            print("You pressed q: Quit")
            return 'q'

# Function to append data to a list for saving
def save_data_entry(data_list, image_number, caption):
    image_name = f"image_{image_number}"  # Generate a unique image name or ID
    data_list.append({
        'image_index': image_number,
        'image_name': image_name,
        'caption': caption
    })

# Loop through the assigned section of the dataset starting from the last checkpoint
for index in range(last_index, end_index):
    image_number = index
    caption = dataset[image_number]['text']  # Assuming 'text' is the caption column

    user_response = display_image_with_caption(image_number, caption)

    # Image window needs to be closed manually or it will remain open.
    # Image.show() does not allow us to programmatically close the window.

    if user_response == 'q':
        # Save progress (good data, bad data, and checkpoint)
        df_good_data = pd.DataFrame(good_data)
        df_bad_data = pd.DataFrame(bad_data)

        df_good_data.to_csv(good_data_csv, index=False)
        df_bad_data.to_csv(bad_data_csv, index=False)

        # Save the checkpoint (last processed index)
        with open(checkpoint_file, 'w') as f:
            f.write(str(index))

        print("Progress saved. You can resume from this point next time.")
        break
    elif user_response == '1':
        save_data_entry(good_data, image_number, caption)
    else:
        save_data_entry(bad_data, image_number, caption)

# If the loop finishes without quitting, save all data and reset the checkpoint
if user_response != 'q' and user_response is not None:
    df_good_data = pd.DataFrame(good_data)
    df_bad_data = pd.DataFrame(bad_data)

    df_good_data.to_csv(good_data_csv, index=False)
    df_bad_data.to_csv(bad_data_csv, index=False)

    # Save the final index as a checkpoint
    with open(checkpoint_file, 'w') as f:
        f.write(str(end_index))

    print("All data processed and saved.")
