import pandas as pd
import sqlite3
import numpy as np
from tqdm.auto import tqdm
import spacy
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
import torch


def generate_csv(input_db_path, db_type, output_path):
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    tqdm.pandas()  # Enable tqdm for pandas

    # Load the data based on db_type
    if db_type == "sql":
        with sqlite3.connect(input_db_path) as con:
            df = pd.read_sql('select * from captions', con)
    elif db_type == "csv":
        df = pd.read_csv(input_db_path)
        df["cap"] = df["generated_caption"]  # Assuming 'generated_caption' exists in the CSV
    else:
        assert False, "db_type must be either 'sql' or 'csv'"

    # # Apply the function with progress tracking
    # df[['verbs', 'nouns', 'adjs']] = df['cap'].progress_apply(lambda x: pd.Series(get_verbs_objects_adjectives(x)))

    # Create the _n columns by calculating the length of each set
    df['verb_n'] = df['verb_list'].apply(len)
    df['obj_n'] = df['obj_list'].apply(len)
    # df['adj_n'] = df['adjs'].apply(len)

    # Define a function to extract the subject using spaCy
    def extract_subject(caption):
        doc = nlp(caption)
        for token in doc:
            if token.dep_ == 'nsubj':  # Find the nominal subject
                return token.text
        return None  # If no subject is found, return None

    # Apply the function to the 'caption' column and create a new 'subject' column
    # df['subject'] = df['caption'].apply(extract_subject)
    df['subject'] = df['caption'].progress_apply(extract_subject)

    # Save the resulting dataframe to a CSV file
    df.to_csv(output_path, index=False)

def dilute_df(df, filter_elem, max_repetitions=5, min_count=1):
    # Create lists to store final, diluted rows and track indices
    diluted_rows = []
    sampled_indices = set()
    diluted_elems = set()  # Track elements that have been diluted

    # Count appearance of each element
    count_per_elem = pd.Series(e for e_list in df[f'{filter_elem}_list'] for e in e_list).value_counts()

    # Loop through each element and its count
    for elem, count in count_per_elem.items():
        # Only process elements with count > min_count (e.g., 4)
        if count <= min_count:
            continue

        # Get rows that contain this element
        rows_with_elem = df[df[f'{filter_elem}_list'].apply(lambda x: elem in x)]

        # Filter out rows that have already been sampled
        rows_with_elem = rows_with_elem[~rows_with_elem.index.isin(sampled_indices)]

        # Check for conjugation with previously diluted elements
        rows_with_elem = rows_with_elem[rows_with_elem[f'{filter_elem}_list'].apply(
            lambda elem_list: not any(e in diluted_elems for e in elem_list)
        )]

        # If the element appears more than the allowed max_repetitions, sample max_repetitions rows
        if len(rows_with_elem) > max_repetitions:
            sampled_rows = rows_with_elem.sample(n=max_repetitions, random_state=42)
        else:
            sampled_rows = rows_with_elem

        # Add sampled rows to the final list
        diluted_rows.append(sampled_rows)

        # Track the indices of the rows that have been sampled
        sampled_indices.update(sampled_rows.index)

        # Add the current element to the set of diluted elements
        diluted_elems.add(elem)

    # Concatenate the diluted rows back into a final DataFrame
    df_diluted = pd.concat(diluted_rows).reset_index(drop=True)

    return df_diluted

# Process the DataFrame and apply LLM logic with one LLM call
def process_caption(row, pipe):
    caption = row['caption']
    verbs = row['verb_list']
    nouns = row['obj_list']

    # Prepare a single prompt for all checks
    combined_prompt = (
        "For example: \n"
        "1. Are the verbs {'harvesting', 'extracting', 'bottling'} in this sentence 'A beekeeper harvesting, extracting, bottling honey from their hives.' synonymous or have a parallel meaning?\n"
        "2. Are all the verbs {'harvesting', 'extracting', 'bottling'} in this sentence 'A beekeeper harvesting, extracting, bottling honey from their hives.' visual verbs that can be perceived from an image?\n"
        "3. Which noun from {'hives', 'beekeeper', 'honey'} is performing the actions {'harvesting', 'extracting', 'bottling'} in this sentence 'A beekeeper harvesting, extracting, bottling honey from their hives.'?\n"
        f"1. no\n"
        f"2. yes\n"
        f"3. beekeeper\n"
        "For example: \n"
        "1. Are the verbs {'marking', 'chiming', 'tick'} in this sentence 'The old clock ticking, chiming, and marking time in the antique store.' synonymous or have a parallel meaning?\n"
        "2. Are all the verbs {'marking', 'chiming', 'tick'} in this sentence 'The old clock ticking, chiming, and marking time in the antique store.' visual verbs that can be perceived from an image?\n"
        "3. Which noun from {'clock', 'store'} is performing the actions {'marking', 'chiming', 'tick'} in this sentence 'The old clock ticking, chiming, and marking time in the antique store.'?\n"
        f"1. no\n"
        f"2. no\n"
        f"3. clock\n"
        "For example: \n"
        "1. Are the verbs {'shapes', 'trims', 'prunes'} in this sentence 'A gardener prunes, trims, and shapes shrubs in a lush garden.' synonymous or have a parallel meaning?\n"
        "2. Are all the verbs {'shapes', 'trims', 'prunes'} in this sentence 'A gardener prunes, trims, and shapes shrubs in a lush garden..' visual verbs that can be perceived from an image?\n"
        "3. Which noun from {'gardener', 'shrubs', 'garden'} is performing the actions {'shapes', 'trims', 'prunes'} in this sentence 'A gardener prunes, trims, and shapes shrubs in a lush garden..'?\n"
        f"1. yes\n"
        f"2. yes\n"
        f"3. gardener\n"
        "For example: \n"
        "1. Are the verbs {'honking', 'cackling', 'quacking'} in this sentence 'A group of ducks honking, cackling, and quacking joyfully.' synonymous or have a parallel meaning?\n"
        "2. Are all the verbs {'honking', 'cackling', 'quacking'} in this sentence A group of ducks honking, cackling, and quacking joyfully.' visual verbs that can be perceived from an image?\n"
        "3. Which noun from {'group', 'ducks'} is performing the actions {'honking', 'cackling', 'quacking'} in this sentence 'A group of ducks honking, cackling, and quacking joyfully.'?\n"
        f"1. yes\n"
        f"2. no\n"
        f"3. ducks\n"
        f"Now answer:\n"
        f"1. Are the verbs {verbs} in this sentence '{caption}' synonymous or have a parallel meaning? "
        f"Answer only 'yes' or 'no'.\n"
        f"2. Are all the verbs {verbs} in this sentence '{caption}' visual verbs that can be perceived from an image? "
        f"Answer only 'yes' or 'no'.\n"
        f"3. Which noun from {nouns} is performing the actions {verbs} in this sentence '{caption}'? "
        f"Answer with only one noun.\n"
        f"No explanation, only one word answers:\n"
        f"1. "
    )

    # Query the LLM with a single prompt and set `max_new_tokens`
    response = pipe(combined_prompt, max_new_tokens=50, num_return_sequences=1)[0]['generated_text']

    # Parse the response by splitting into lines and extracting the first word from each answer
    lines = response[len(combined_prompt)-3:].strip().split('\n')
    synonym_response = lines[0].split()[-1].lower()  # Extract the first word from the synonym response
    visual_response = lines[1].split()[-1].lower()   # Extract the first word from the visual verb response
    subject_noun_response = lines[2].split()[-1].lower()  # Extract the first word from the noun performing the action

    return synonym_response, visual_response, subject_noun_response


def process_in_batches(df, batch_size, process_func, output_path, pipe):
    """
    Process a DataFrame in batches and save results periodically with progress tracking.

    Parameters:
    - df: DataFrame to process
    - batch_size: Number of rows per batch
    - process_func: Function to apply to each row
    - output_path: Path to save the results
    """
    total_rows = len(df)
    # Ensure new columns exist in the DataFrame, initialized with None or NaN
    df['synonym_check'] = None
    df['visual_check'] = None
    df['subject_noun'] = None

    # Initialize progress bar
    with tqdm(total=total_rows, desc="Processing rows") as pbar:
        start_row = 0

        while start_row < total_rows:
            end_row = min(start_row + batch_size, total_rows)
            chunk_df = df.iloc[start_row:end_row].copy()  # Create a copy to avoid SettingWithCopyWarning

            # Apply the processing function
            results = chunk_df.apply(
                lambda row: process_func(row, pipe),
                axis=1,
                result_type="expand"
            )

            # Assign results to the copy
            df.iloc[start_row:end_row, df.columns.get_loc('synonym_check'):df.columns.get_loc('subject_noun')+1] = results.values
            # chunk_df.loc[start_row:end_row, ['synonym_check', 'visual_check', 'subject_noun']] = results

            # Append to the output CSV
            # df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
            df.to_csv(output_path, index=False)

            # Update progress bar
            pbar.update(end_row - start_row)

            # Move to the next batch
            start_row = end_row


def filter_csv(cache_dir, input_path, folder_name, model_name, token, llm_fil_flag):
    if llm_fil_flag==True:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        eos_token_id = tokenizer.eos_token_id

        # Load in 4bit config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        # Load mistral 7B model with 4bit config
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=token,
            cache_dir=cache_dir  # Model cache directory (model takes up ~15G in location)
        )

        # Pipe-line of text-generation with preloaded model and tokenizer
        # Hagar: I needed to update- bit and bites > 0.41.3
        pipe = pipeline(
            "text-generation",
            batch_size=8,
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            device_map="auto",
            pad_token_id=eos_token_id,  # Explicitly setting pad_token_id
        )

    df = pd.read_csv(input_path)

    # Apply basic filters
    fl_df = df.drop_duplicates(subset=['caption']).reset_index(drop=True)

    # Add column for verb, object, adjective count
    for col in ['verb', 'obj']:
        fl_df  = fl_df[fl_df[f'{col}_list'].str.endswith('}')].reset_index(drop=True)  # Filter out rows where the cell does not end with '}'
        fl_df[f'{col}_list'] = fl_df[f'{col}_list'].apply(
            lambda x: set(item.strip().strip("'") for item in x.strip('{}').split(','))
        )
        # df[f'{col}_list'] = df[f'{col}_list'].apply(lambda x: ast.literal_eval(x))
        fl_df[f'{col}_n'] = fl_df[f'{col}_list'].str.len()  # counts 'verb' instances

    fl_df = fl_df[fl_df['verb_n'].between(2, 6)]  # Require 3-6 present verbs
    fl_df = fl_df[fl_df['caption'].str.endswith('.')]  # Require '.' at the end of the sentence

    o_diluted_df = dilute_df(fl_df, filter_elem='obj', max_repetitions=6)
    final_diluted_df = dilute_df(o_diluted_df, filter_elem='verb', max_repetitions=8)
    final_diluted_df = final_diluted_df.drop_duplicates(subset='caption').reset_index(drop=True)

    # Initialize tqdm for pandas
    tqdm.pandas()

    # fl_df = fl_df.iloc[:20]
    if llm_fil_flag == True:
        # Prepare the path and other parameters
        llm_check_path = os.path.join(folder_name, 'llm_check_df.csv')
        batch_size = 500

        # Call the function
        process_in_batches(final_diluted_df, batch_size, process_caption, llm_check_path, pipe)

        llm_check_df = pd.read_csv(llm_check_path).reset_index(drop=True)

    llm_fl_df = llm_check_df[(llm_check_df['synonym_check'] == 'yes') & (llm_check_df['visual_check'] == 'yes')]
    llm_filtered_path = os.path.join(folder_name, 'filtered_verbs_llm.csv')

    final_diluted_df = llm_fl_df.sample(frac=1).reset_index(drop=True)
    final_diluted_df.to_csv(llm_filtered_path, index=False)


if __name__ == "__main__":
    """
    generate_csv function: Takes as input the database of captions (in this case generated by Mistral), and converts it to a CSV file 
    with the verbs extracted from each caption
    filter_csv function: Filters the CSV that generate_csv outputs to only keep 'good' captions
    """
    # Get the current working directory
    folder_name = 'data'
    db_dir = os.path.join(folder_name, 'verb_synthetic_captions.db')
    unfiltered_csv_dir = os.path.join(folder_name, 'unfiltered_verbs.csv')
    generate_csv(
        input_db_path=db_dir,
        db_type='sql',
        output_path=unfiltered_csv_dir
    )

    cache_dir = os.path.join(os.path.dirname(os.getcwd()), 'models', 'cache_dir')
    model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
    token = 'your_hf_token'
    filter_csv(
        cache_dir=cache_dir,
        input_path=unfiltered_csv_dir,
        folder_name=folder_name,
        model_name=model_name,
        token=token,
        llm_fil_flag=True  # Choose if to implement llm filter (visual+synonym)
    )
