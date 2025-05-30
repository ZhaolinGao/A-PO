# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd

import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 beta1=0.5,
                 filter_incorrect=False,
                 num_gen_to_use=32,
                 max_prompt_length=1024,
                 add_prompt_template=True,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error'):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.beta1 = beta1
        self.filter_incorrect = filter_incorrect
        self.num_gen_to_use = num_gen_to_use
        self.max_prompt_length = max_prompt_length
        self.add_prompt_template = add_prompt_template
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self._download()
        self._read_files_and_tokenize()
        self._process_values()
        self._filter_dataset()

    def _download(self):
        from verl.utils.fs import copy_local_path_from_hdfs
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:

            # load huggingface dataset
            if not parquet_file.endswith('.parquet'):
                dataframe = load_dataset(parquet_file, split='train').to_pandas()
            # read parquet files and cache
            else:
                dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        if self.filter_prompts:
            self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
                tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                                                                axis=1)]

        print(f'filter dataset len: {len(self.dataframe)}')

    def _process_values(self):

        columns = self.dataframe.columns.tolist()

        if self.beta1 < 0 or 'eval_0' not in columns:
            v_star = np.zeros(len(self.dataframe))
            print("v_star is set to 0")
        elif self.beta1 == 0:
            selected_cols = [f"eval_{i}" for i in range(self.num_gen_to_use)]
            v_star = self.dataframe[selected_cols].to_numpy()
            v_star = np.mean(v_star, axis=1)
            print(f"v_star is set to mean with num_gen_to_use = {self.num_gen_to_use}")
        else:
            selected_cols = [f"eval_{i}" for i in range(self.num_gen_to_use)]
            v_star = self.dataframe[selected_cols].to_numpy()
            v_star = np.exp(v_star / self.beta1).mean(axis=1)
            v_star = np.log(v_star) * self.beta1
            print(f"v_star is set to exp with num_gen_to_use = {self.num_gen_to_use}, beta1 = {self.beta1}")

        self.dataframe['v_star'] = list(v_star)

    def _filter_dataset(self):

        if self.filter_incorrect:
            print(f"original dataset len: {len(self.dataframe)}")
            assert self.num_gen_to_use > 0
            selected_cols = [f"eval_{i}" for i in range(self.num_gen_to_use)]
            evals = self.dataframe[selected_cols].to_numpy()
            evals = np.mean(evals, axis=1)
            self.dataframe = self.dataframe[evals != 0]
            print(f'after filtering prompts with all incorrect responses: {len(self.dataframe)}')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        prompt_with_chat_template = chat[0]['content']
        # prompt_with_chat_template = chat

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         add_prompt_template=self.add_prompt_template,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["v_star"] = torch.tensor(row_dict["v_star"], dtype=torch.float32)

        return row_dict
