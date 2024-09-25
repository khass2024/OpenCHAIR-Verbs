import pandas as pd
from tqdm.auto import tqdm
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch
import sqlite3
import logging
import traceback
import os


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


def generate_claude_based_captions(model_name, token, cache_dir, gpt_csv_dir, dil_claude_csv_dir, db_dir, caption_n):
    """
    Generates the DB with hopefully huge amount of captions which will later be filtered to extract 'good' captions
    """
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
        # batch_size=16,
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="auto",
        pad_token_id=eos_token_id,  # Explicitly setting pad_token_id
    )

    df = pd.read_csv(gpt_csv_dir).rename(
        columns={
            'Caption': 'cap',
            'verbs': 'v_list',
            'nouns': 'o_list',
            'adjectives': 'a_list'
        }
    )
    df = df.iloc[::-1].reset_index(drop=True)  # Reverse order to get other selection
    print(f'Original dataframe rows: {len(df)}')

    # Add column for verb, object, adjective count
    for col in ['v', 'o', 'a']:
        df[f'{col}_list'] = df[f'{col}_list'].apply(lambda x: set(item.strip() for item in x.strip('{}').split(',')))
        df[f'{col}_n'] = df[f'{col}_list'].str.len()  # counts 'verb' instances

    o_diluted_df = dilute_df(df, filter_elem='o', max_repetitions=3)  # Each object should repeat 3 times or less
    # final_diluted_df = o_diluted_df
    final_diluted_df = dilute_df(o_diluted_df, filter_elem='v', max_repetitions=7)  # Each verb should repeat 7 times or less
    final_diluted_df = final_diluted_df.drop_duplicates(subset='caption').reset_index(drop=True)
    final_diluted_df.to_csv(dil_claude_csv_dir, index=False)  # save the csv each 1000 iterations
    print(f'Diluted dataframe rows: {len(final_diluted_df)}')

    # Connect to the SQLite database
    con = sqlite3.connect(db_dir)
    cur = con.cursor()

    # Create table to store captions
    cur.execute('''
        CREATE TABLE IF NOT EXISTS captions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            caption TEXT,
            verb_list TEXT,
            obj_list TEXT,
            subject TEXT
        )
    ''')
    # adj_list TEXT
    con.commit()

    def get_prompt(K=50):
        """
        Generates a prompt with captions and verb lists.
        """
        subdf = final_diluted_df.sample(K)
        out = (
            'Generate a diverse and creative caption similar to those in image datasets like MSCOCO and Flickr. '
            'The caption should contain action-oriented verbs. Enhance the caption by incorporating synonyms for the main verb. '
            'Ensure the diversity in verbs comes between captions, with no repetition across captions. '
            'Promote variety in subjects, including animals, objects, and different scenes, while avoiding overly repetitive human-centered actions. '
            'Using basic terms like man, woman, child, or people is fine if needed, and preferable to repeating specific professions. '
            'Each caption must describe a new and distinct scene, with no repeated scenarios or actions within or between captions.'
            'Avoid beekeeper, beaver, astronaut, potter, magician, sculptor, lighthouse. Add man, women, child, people, Fantasy characters, machines, nature phenomena'
        )

        for row in subdf.itertuples():
            out += f"Caption: {row.caption}\nVerbs: {row.v_list}\nNouns: {row.o_list}\nSub: {row.sub}\n"  # Adjectives: {row.a_list}\n
        return out

    # Function to count current entries in the database
    def get_current_count():
        cur.execute("SELECT COUNT(*) FROM captions")
        return cur.fetchone()[0]

    generated_count = get_current_count()

    # Progress bar to track the number of captions generated
    pbar = tqdm(total=caption_n)

    while generated_count < caption_n:
        # Generate prompt and captions
        prompt = get_prompt()
        out = pipe(prompt, max_new_tokens=300, temperature=1.1, top_k=50, num_return_sequences=1, do_sample=True)

        for x in out:
            text_list = x['generated_text'][len(prompt):].strip().split('\n')

            # Ensure the structure is valid
            current_captions = []
            elem_n = 4
            for caption_i in range(len(text_list) // elem_n):
                if all(['Caption: ' in text_list[caption_i * elem_n + 0],
                        'Verbs: ' in text_list[caption_i * elem_n + 1],
                        'Nouns: ' in text_list[caption_i * elem_n + 2],  # ,'Adjectives: ' in text_list[3]
                        'Sub: ' in text_list[caption_i * elem_n + 3]]):  # ,'Adjectives: ' in text_list[3]

                    # Check that the last noun list ends with '}'
                    # last_noun_list = text_list[caption_i * elem_n + 2].replace('Nouns: ', '').strip()
                    last_sub_list = text_list[caption_i * elem_n + 3].replace('Sub: ', '').strip()
                    if last_sub_list.endswith('}'):
                        caption = text_list[caption_i * elem_n + 0].replace('Caption: ', '').strip()
                        verb_list = text_list[caption_i * elem_n + 1].replace('Verbs: ', '').strip()
                        obj_list = text_list[caption_i * elem_n + 2].replace('Nouns: ', '').strip()
                        sub = last_sub_list

                        # Insert into the database
                        cur.execute('''
                            INSERT INTO captions (caption, verb_list, obj_list, subject)
                            VALUES (?, ?, ?, ?)
                        ''', (caption, verb_list, obj_list, sub))
                        con.commit()

                        # Update the count of generated captions
                        generated_count += 1
                        current_captions.append(caption)

                        # Update the progress bar
                        pbar.update(1)

        # Check if we've reached the desired number of captions
        if generated_count >= caption_n:
            break

    # Close the progress bar and database connection
    pbar.close()
    con.close()

if __name__ == "__main__":
    # Example usage: customize model_name, token, and caption_n
    model_name = 'mistralai/Mistral-7B-Instruct-v0.2'  #
    token = 'your_hf_token'
    cache_dir = os.path.join(os.path.dirname(os.getcwd()), 'models', 'cache_dir')
    folder_name = 'data'
    claude_csv_dir = os.path.join(folder_name, 'claude_verb_seed.csv')
    dil_claude_csv_dir = os.path.join(folder_name, 'diluted_claude_verb_seed.csv')  # Rare claude dataset which generation 'sees'
    csv_dir = os.path.join(folder_name, 'verb_synthetic_captions.db')  # Assign new database name
    caption_n = 25000  # Required Caption num

    generate_claude_based_captions(model_name, token, claude_csv_dir, dil_claude_csv_dir, csv_dir, caption_n)
