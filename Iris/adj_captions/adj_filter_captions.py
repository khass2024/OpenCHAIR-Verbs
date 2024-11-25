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

# python -m spacy download en_core_web_trf
nlp = spacy.load("en_core_web_trf")

@lru_cache(maxsize=None)
def get_verbs(cap):
    doc = nlp(cap.lower())
    return {token.lemma_ for token in doc if token.pos_ == 'VERB'}

def get_adj(cap):
    doc = nlp(cap.lower())
    return {token.lemma_ for token in doc if token.pos_ == 'ADJ'}

@lru_cache(maxsize=None)
def get_base_verbs(cap):
    doc = nlp(cap.lower())
    return {token.lemma_ for token in doc}

@lru_cache(maxsize=None)
def cap2objs(cap):
    verbs = get_verbs(cap)
    return {v for v in verbs}

def cap2adjs(cap):
    adjs = get_adj(cap)
    return {a for a in adjs}


def visual_check_prompt(sentence, word):
    return f'''I will provide a sentence, and a word that exists in the sentence. 
    My question to you is whether this word should be tagged as a visual adjective or a non-visual different adjective in the sentence.
    Visual adjectives describe attributes that can be perceived visually from an image (cannot be perceived auditorily or by sense of smell).
    Ignore grammatical issues or whether the described situation is possible or not. Only whether it is a visual adjective or not in the context of the provided sentence.
    Provide an answer as follows: 
    Provide an explanation for your answer and whether you think the word is a visual / non-visual
    Do it step by step, and finally provide the final answer as yes / no.
    Restrict yourself to 50 tokens-length explanation.
    Here are a few examples:
    sentence: The yellow sun was shining.
    word: yellow
    explanation: yellow is a visual adjective because it describes a color that can be seen in an image. Answer: yes
    sentence: The loud music was playing.
    word: loud
    explanation: loud is an auditory adjective and cannot be perceived visually from an image. Answer: no
    sentence: A silent blue swing in the park.
    word: silent
    explanation: blue is a visual adjective, but silent is not. Answer: unsure
    sentence: a Three raindrops cascade off a curved leaf.
    word: curved
    explanation: curved is a visual attribute. Answer: yes
    sentence: a horse drinking cold coffee
    word: cold
    explanation: can't tell if coffee is hot by looking at a picture of it, hot is not visual. Answer: no
    sentence: a mouse eating stringy mozzarella
    word: stringy
    explanation: stringy can be seen visually. Answer: yes
    sentence: chickens clucking loudly in the yard.
    word: loudly
    explanation: this is an auditory description which we couldn't tell from an image, not visual. Answer: no
    sentence: Moonlight illuminates a forest floor peppered with fallen leaves.
    word: fallen
    explanation: Fallen leaves can be seen visually. Answer: yes
    sentence: a tiger eating a stripy zebra
    word: stripy
    explanation: stripy is a visual attribute. Answer: yes
    sentence: sliced beets on a smelly cutting board.
    word: smelly
    explanation: the word "smelly" functions as a non-visual adjective, it can't be surmised from an image. Answer: no
    sentence: {sentence}
    word: {word}
    explanation: '''

def adj_visual_check(pipe, sentence, adjs):
    visual_tags = []
    explanations = []
    for adj in list(adjs):
        prompt = visual_check_prompt(sentence, adj)
        result = pipe(prompt, max_new_tokens=50, temperature=0.8, num_return_sequences=1, do_sample=True)

        output_text = result[0]['generated_text'].strip().lower()
        explanation = output_text.split(f'word: {adj}')[-1].strip() if f'word: {adj}' in output_text else 'No explanation'
        explanation = explanation.split('\n')[0].strip() if '\n' in explanation else explanation
        if 'answer: yes' in explanation:
            visual_tag = 'yes'
        elif 'answer: no' in explanation:
            visual_tag = 'no'
        elif 'answer: unsure' in explanation:
            visual_tag = 'unsure'
        else:
            visual_tag = 'Nan'

        visual_tags.append(visual_tag)
        explanations.append(explanation)

    return visual_tags, explanations


def generate_csv(input_db_path, db_type, output_path):
    tqdm.pandas()

    if db_type == "sql":
        # load if db is sql
        with sqlite3.connect(input_db_path) as con:
            df = pd.read_sql('select * from data', con)
    elif db_type == "csv":
        # load if db is csv
        df = pd.read_csv(input_db_path)
        df["cap"] = df["generated_caption"]
    else:
        assert False

    ## Hagar's Version:
    df['adjs'] = df.cap.progress_apply(get_adj)  # use spacy to recognize adjectives in captions
    df['n_adjs'] = df.adjs.str.len()

    df.to_csv(output_path)

    return df

def filter_adjs(row):
    # Extract 'adjs' values where corresponding 'adj_visual_pass' value is 'yes'
    filtered_adjs = [adj for adj, pass_ in zip(row['adjs'], row['adj_visual_pass']) if pass_ == 'yes']
    # Return filtered 'adjs' if there's at least one 'yes'
    return filtered_adjs if filtered_adjs else None


def filter_csv(cache_dir, input_path, output_path, filter_type, model_name, token):
    # Use AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure padding token is the same as eos
    tokenizer.pad_token = "[PAD]"  # Optional: assign a string for clarity
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto',
        cache_dir=cache_dir
    )

    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map='auto',
        pad_token_id=tokenizer.eos_token_id,
        batch_size=32
    )

    # filter_type = 'rare' / integer
    df = pd.read_csv(input_path)
    columns_to_keep = ['cap', 'adjs', 'n_adjs']

    df = df[df['n_adjs'] > 0]  # drop caption with no spacy adj_captions
    df = df[df['n_adjs'] <= 3]  # drop caption with over 3 spacy adj_captions

    df = df[df.cap.str.endswith('.')]  # drop caption with no '.' end of sentence
    df.adjs = df.adjs.apply(eval)

    # df = df[df.adjs.apply(lambda x: all(len(y) > 3 for y in x))]  # Ignore short verbs due to spacy errors
    adjs_counts = pd.Series(x for row in df.adjs for x in row).value_counts()  # count

    if filter_type == 'rare':
        S = adjs_counts.sum()

        rare_adjs = adjs_counts[adjs_counts.cumsum() > 0.85 * S].copy()

        df_rare = df[df.adjs.apply(lambda x: all(y in rare_adjs.index for y in x))].copy()
        df_rare.reset_index(drop=True, inplace=True)
        df_rare = df_rare[columns_to_keep]
    else:
        filter_val = int(filter_type)

        adjs_set = set(adjs_counts.index)
        df_rare = pd.DataFrame()
        for adj in adjs_set:
            df2 = df[df.adjs.apply(lambda x: adj in x)]
            df = df[df.adjs.apply(lambda x: adj not in x)]

            rand_ind = np.random.permutation(len(df2))

            df_rare = pd.concat([df_rare, df2.iloc[rand_ind[:filter_val]]])

    # Check each caption with adj_visual_check
    tqdm.pandas()  # Adding progress bar to the adj_visual_check step

    # Apply the function and split the results into two columns
    df_rare[['adj_visual_pass', 'adj_explanation']] = df_rare.progress_apply(
        lambda row: pd.Series(adj_visual_check(pipe, row['cap'], row['adjs'])),
        axis=1
    )
    df_rare.to_csv('data_adj/adj_trial_vis.csv')

    # Apply the function to filter 'adjs' and create a new column 'filtered_adjs'
    df_rare['filtered_adjs'] = df_rare.apply(filter_adjs, axis=1)

    # Filter rows where 'filtered_adjs' is not None
    df_rare = df_rare[df_rare['filtered_adjs'].notnull()]

    # Recalculate the number of adjectives (n_adj) in the filtered DataFrame
    df_rare['n_adj'] = df_rare['filtered_adjs'].apply(len)

    # df_rare = df_rare[df_rare['adj_visual_pass'] == 'yes']  # save captions passing the 'visual check'
    df_rare.reset_index(drop=True, inplace=True)

    df_rare = df_rare.drop_duplicates(subset=['cap'])
    df_rare.to_csv(output_path)

    return df_rare

if __name__ == "__main__":
    """
    generate_csv function: Takes as input the database of captions (in this case generated by Mistral), and converts it to a CSV file 
    with the adjectives extracted from each caption
    filter_csv function: Filters the CSV that generate_csv outputs to only keep 'good' captions
    """
    # Get the current working directory
    folder_name = 'data'
    db_dir = os.path.join(folder_name, 'adj_synthetic_captions.db')
    unfiltered_csv_dir = os.path.join(folder_name, 'unfiltered_adj.csv')
    filtered_csv_dir = os.path.join(folder_name, 'filtered_adj_visual.csv')
    unfil_df = generate_csv(
        input_db_path=db_dir,
        db_type='sql',
        output_path=unfiltered_csv_dir
    )

    cache_dir = os.path.join(os.path.dirname(os.getcwd()), 'models', 'cache_dir')
    model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
    token = 'your_hf_token'
    login(token=token)
    fil_df = filter_csv(
        cache_dir=cache_dir,
        input_path=unfiltered_csv_dir,
        output_path=filtered_csv_dir,
        filter_type='rare',
        model_name=model_name,
        token=token
    )
