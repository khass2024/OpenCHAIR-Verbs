from tqdm import tqdm
import pandas as pd
from diffusers import DiffusionPipeline
import torch
import os
import argparse

SD_KWARGS = {
    'guidance_scale': 10,
    'num_inference_steps': 40,
    'negative_prompt': "unclear, deformed, out of image, disfiguired, body out of frame"
}

def generate_dataset(args):
    data = pd.read_csv(args.data_file)
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", cache_dir=args.cache_dir)
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    BS = args.batch_size
    data_ = data.set_index(data.index // BS).copy() 

    sd_kwargs = SD_KWARGS.copy()
    sd_kwargs["negative_prompt"] = [sd_kwargs["negative_prompt"]] * BS

    start_idx = 0  # 798 (why???) maybe this was a memory issue
    j = start_idx
    for i in tqdm(range(start_idx, data_.index.max()+1)):
        if BS == 1:
            prompts = [data_.loc[i].caption]  # generated_caption
        else:
            prompts = data_.loc[i].caption.tolist()  # generated_caption
            
        out = pipe(
            prompts,
            **sd_kwargs,
            seed=args.seed,
        )
        for img in out.images:
        #for img, img_id in zip(out.images, image_ids):
            fn = args.output_dir + str(j).rjust(6, '0') + '.jpg'
            img.save(fn)
            j += 1

        del out
        torch.cuda.empty_cache()  # Free memory after each batch


if __name__ == "__main__":
    """
    Generates the images from input CSV with captions
    """
    folder_name = 'verb_captions'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default='stabilityai/stable-diffusion-xl-base-1.0')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default='models/cache_dir')  # Model cache directory (model takes up ~7-8GB in location)
    parser.add_argument("--data-file", type=str, default=os.path.join(folder_name, 'data', 'filtered_verbs_llm.csv'))
    parser.add_argument("--output-dir", type=str, default=os.path.join(folder_name, 'data', 'images/'))
    args = parser.parse_args()

    generate_dataset(args)
