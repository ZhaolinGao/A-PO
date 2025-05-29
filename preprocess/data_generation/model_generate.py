from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from verl.utils.reward_score import gsm8k, math

import argparse
import os
import torch
import random
import numpy as np


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _select_rm_score_fn(reward_function):
    if reward_function == 'gsm8k':
        return gsm8k.compute_score
    elif reward_function == 'math':
        return math.compute_score_preprocessing
    else:
        raise NotImplementedError


def evaluate(dataset, reward_name, reward_function, n):

    all_scores = []

    for i in tqdm(range(n)):
        scores = []
        for d in range(len(dataset)):
            if reward_name == 'gsm8k':
                _, cs = reward_function(dataset[d][f'response_{i}'], dataset[d]['reward_model']['ground_truth'], method='flexible')
            else:
                _, cs = reward_function(dataset[d][f'response_{i}'], dataset[d]['reward_model']['ground_truth'])
            scores.append(cs)
        scores = np.array(scores)
        all_scores.append(scores)
        dataset = dataset.add_column(f"eval_{i}", scores)

    return dataset


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dataset", type=str, default="~/data/gsm8k/train.parquet")
    parser.add_argument("--add_prompt_template", type=bool, default=False)
    parser.add_argument("--remote_dir", type=str, default=None, required=True)
    parser.add_argument("--reward_function", type=str, default="gsm8k")
    parser.add_argument("--maxlen", type=int, default=1024)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--world_size", type=int, default=1)
    return parser.parse_args()


def main():

    # init
    args = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.world_size,
    )

    # dataset
    dataset = load_dataset("parquet", data_files={'train': args.dataset}, split='train')
    if args.end_idx != -1:
        dataset = dataset.select(range(args.start_idx, args.end_idx))

    # preprocess dataset
    if args.add_prompt_template:
        prompts = [tokenizer.apply_chat_template(dataset[i]['prompt'], add_generation_prompt=True, tokenize=False) for i in tqdm(range(len(dataset)))]
    else:
        prompts = [dataset[i]['prompt'][0]['content'] for i in range(len(dataset))]

    # start generate
    for p in range(args.n):
        set_seed(p * 50)
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=args.maxlen,
            seed=p * 50,
        )
        response = llm.generate(prompts, sampling_params)
        output = list(map(lambda x: x.outputs[0].text, response))
        dataset = dataset.add_column(f"response_{p}", output)

    # reward function
    reward_function = _select_rm_score_fn(args.reward_function)

    # evaluate
    dataset = evaluate(dataset, args.reward_function, reward_function, args.n)

    # save
    dataset.push_to_hub(args.remote_dir)


if __name__ == "__main__":
    main()