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

def generate_claude_based_captions(model_name, token, cache_dir, gpt_csv_dir, db_dir, caption_n):
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
        cache_dir=cache_dir  # "models/cache_dir"  # Model cache directory (model takes up ~15G in location)
    )

    # Pipe-line of text-generation with preloaded model and tokenizer
    pipe = pipeline(
        "text-generation",
        # batch_size=16,
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="auto",
        pad_token_id=eos_token_id,  # Explicitly setting pad_token_id
        # max_new_tokens=100
    )

    df = pd.read_csv(gpt_csv_dir).rename(
        columns={
            'Caption': 'cap',
            'Visual Attributes': 'n'
        }

    )
    df.n = df.n.apply(eval)  # eval, converts the string into a Python set object
    df['nn'] = df.n.str.len()  # counts 'n' instances (is n verbs? currently it includes objs, filtered later?)

    n = pd.Series(x for row in df.n for x in row).value_counts()

    S = n.sum()  # sum all 'verbs' (not actually verbs)

    rare_n = n[n.cumsum() > 0.50 * S].copy()  # filter out if over 0.5 of verbs

    df_rare = df[(df.nn > 0) & (df.n.apply(lambda x: all(y in rare_n.index for y in x)))].copy()

    def get_prompt(K=10):
        subdf = df_rare.sample(K)
        out = ''
        for row in subdf.itertuples():
            # out += str(row.n) + ' => ' + row.cap + '\n'
            out += row.cap + '\n'
        return out


    # Database connection and cursor
    # Remove existing database file if it exists
    if os.path.exists(db_dir):
        os.remove(db_dir)

    con = sqlite3.connect(db_dir)
    cur = con.cursor()

    # Create table if it doesn't exist
    try:
        cur.execute('''
        CREATE TABLE IF NOT EXISTS data (
            cap TEXT
        )
        ''')
    except Exception as e:
        logging.error(traceback.format_exc())
        print(f"Error creating table: {e}")

    # Insert query
    query = """
    INSERT INTO data (cap)
    VALUES (?)
    """

    # Progress bar
    pbar = tqdm(total=caption_n)

    # Function to count current entries in the database
    def get_current_count():
        cur.execute("SELECT COUNT(*) FROM data")
        return cur.fetchone()[0]

    # Generate captions until there are 'caption_n' elements in the database
    while get_current_count() < caption_n:
        prompt = get_prompt()
        out = pipe(prompt, max_new_tokens=50, temperature=0.8, num_return_sequences=1, do_sample=True)  # hagar added do_sample=True
        for x in out:
            text_list = x['generated_text'][len(prompt):].strip().split('\n')

            for text in text_list:
                data = (text,)
                cur.execute(query, data)
                con.commit()
                pbar.update()

    # Close the progress bar and connection
    pbar.close()
    con.close()

if __name__ == "__main__":
    # Example usage: customize model_name, token, and caption_n
    model_name = 'mistralai/Mistral-7B-Instruct-v0.2'  #
    token = 'your_hf_token'
    cache_dir = os.path.join(os.path.dirname(os.getcwd()), 'models', 'cache_dir')
    folder_name = 'data'
    gpt_csv_dir = os.path.join(folder_name, 'claude_adj_seed.csv')  # Rafi's claude verb captions
    db_dir = os.path.join(folder_name, 'adj_synthetic_captions.db')  # Assign new database name (Adjectives)
    caption_n = 50000  # Required Caption num

    generate_claude_based_captions(model_name, token, cache_dir, gpt_csv_dir, db_dir, caption_n)
