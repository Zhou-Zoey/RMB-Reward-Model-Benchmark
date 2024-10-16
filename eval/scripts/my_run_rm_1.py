# Copyright 2023 AllenAI. All rights reserved.
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

import argparse
import logging
import os
import sys
import yaml
import pandas as pd

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

# from rewardbench import (
#     REWARD_MODEL_CONFIG,
#     check_tokenizer_chat_template,
#     load_eval_dataset,
#     save_to_hub,
# )

from rewardbench import (
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    load_eval_dataset
)

# from utils import load_eval_dataset
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section

# Enable TensorFloat32 (TF32) tensor cores on Ampere GPUs for matrix multiplications (faster than FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer to model")
    parser.add_argument("--dataset_dir", type=str, required=True, help="path to data_dir")
    parser.add_argument("--dataset", type=str, required=True, help="path to data")
    parser.add_argument("--results", type=str, required=True, help="path to results")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for inference")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length of RM inputs (passed to pipeline)")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--debug", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    args = parser.parse_args()
    return args


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_parameters(config, key):
    if key in config:
        return config[key]
    else:
        raise KeyError(f"Configuration for '{key}' not found.")


def main():
    args = get_args()
    ###############
    # Setup logging
    ###############
    accelerator = Accelerator()
    current_device = accelerator.process_index

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    
    p2n = {
        '/home/jovyan/share_fudan/harmless/models/ArmoRM-Llama3-8B-v0.1':'RLHFlow/ArmoRM-Llama3-8B-v0.1',
        '/home/jovyan/share_fudan/harmless/models/Eurus-RM-7b':'openbmb/Eurus-RM-7b',
        '/home/jovyan/share_fudan/harmless/models/stablelm-2-12b-chat':'stabilityai/stablelm-2-12b-chat',
        '/home/jovyan/share_fudan/harmless/models/Starling-RM-34B':'Nexusflow/Starling-RM-34B',
        '/home/jovyan/share_fudan/harmless/models/internlm2-7b-reward':'internlm/internlm2-7b-reward',
        '/home/jovyan/share_fudan/harmless/models/UltraRM-13b':'openbmb/UltraRM-13b',
        '/home/jovyan/share_fudan/harmless/models/PairRM-hf':'llm-blender/PairRM-hf'
    }

    model_config = load_config('/home/jovyan/share_fudan/harmless/reward-bench-new/scripts/configs/eval_configs.yaml')
    model_name = p2n[args.model]
    config_dict = get_parameters(model_config, model_name)
    # load chat template
    if 'chat_template' in config_dict.keys():
        chat_template = config_dict['chat_template']
    else:
        chat_template = "tulu"
    # chat_template = args.chat_template
    try:
        conv = get_conv_template(chat_template)
    except Exception as e:
        conv = get_conv_template("tulu")
    # model_name = p2n[args.model]
    if model_name in REWARD_MODEL_CONFIG:
        config = REWARD_MODEL_CONFIG[model_name]
    else:
        config = REWARD_MODEL_CONFIG["default"]
    logger.info(f"Using reward model config: {config}")

    # Default entries
    # "model_builder": AutoModelForSequenceClassification.from_pretrained,
    # "pipeline_builder": pipeline,
    # "quantized": True,
    # "custom_dialogue": False,
    # "model_type": "Seq. Classifier"

    quantized = config["quantized"]  # only Starling isn't quantized for now
    custom_dialogue = config["custom_dialogue"]
    model_type = config["model_type"]
    model_builder = config["model_builder"]
    pipeline_builder = config["pipeline_builder"]
    torch_dtype = config.get("torch_dtype", None)
    # not included in config to make user explicitly understand they are passing this
    trust_remote_code = args.trust_remote_code

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    # tokenizer_path = args.tokenizer if args.tokenizer else args.model
    if model_name != config_dict['tokenizer']:
        # tokenizer_path = '/mnt/petrelfs/zhengguodong/.cache/huggingface/hub/models--01-ai--Yi-34B-Chat/snapshots/2e528b6a80fb064a0a746c5ca43114b135e30464'
        tokenizer_path = config_dict['tokenizer']
    else:
        tokenizer_path = args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if not custom_dialogue:  # not needed for PairRM / SteamSHP
        tokenizer.truncation_side = "left"  # copied from Starling, but few samples are above context length
    print(args.dataset)
    dataset, subsets = load_eval_dataset(
        core_set=False,
        EXTRA_PREF_SETS = args.dataset_dir,
        conv=conv,
        custom_dialogue_formatting=custom_dialogue,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id"],
    )
    # copy id for saving, then remove
    ids = dataset["id"]
    dataset = dataset.remove_columns("id")

    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(10))
        subsets = subsets[:10]
        ids = ids[:10]

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size
    logger.info("*** Load reward model ***")
    reward_pipeline_kwargs = {
        "batch_size": BATCH_SIZE,  # eval_args.inference_batch_size,
        "truncation": True,
        "padding": True,
        "max_length": args.max_length,
        "function_to_apply": "none",  # Compute raw logits
        "return_token_type_ids": False,
    }
    if quantized:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": {"": current_device},
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {
            "device_map": {"": current_device},
            "torch_dtype": torch_dtype,
        }

    model = model_builder(args.model, **model_kwargs, trust_remote_code=trust_remote_code)
    reward_pipe = pipeline_builder(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
    )

    ############################
    # Tokenization settings & dataset preparation
    ############################
    # set pad token to eos token if not set
    if reward_pipe.tokenizer.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
    # For models whose config did not contains `pad_token_id`
    if reward_pipe.model.config.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

    # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
    if not check_tokenizer_chat_template(tokenizer):
        reward_pipe.tokenizer.add_eos_token = True

    ############################
    # Run inference [1/2]" built in transformers
    ############################
    # if using HF pipeline, can pass entire dataset and get results
    # first, handle custom pipelines that we must batch normally
    if pipeline_builder == pipeline:
        logger.info("*** Running forward pass via built in pipeline abstraction ***")
        # this setup can be optimized slightly with one pipeline call
        # prepare for inference
        reward_pipe = accelerator.prepare(reward_pipe)

        results_rej = reward_pipe(dataset["text_rejected"], **reward_pipeline_kwargs)
        results_cho = reward_pipe(dataset["text_chosen"], **reward_pipeline_kwargs)

        # extract scores from results which is list of dicts, e.g. [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        scores_chosen = [result["score"] for result in results_cho]
        scores_rejected = [result["score"] for result in results_rej]

        # pairwise comparison list comprehension
        results = [1 if chosen > rejected else 0 for chosen, rejected in zip(scores_chosen, scores_rejected)]

    ############################
    # Run inference [2/2] custom pipelines
    ############################
    else:
        logger.info("*** Running dataloader to collect results ***")
        # TODO make more custom pipelines work with pre-tokenized data
        from torch.utils.data.dataloader import default_collate

        # for PairRM, hmm, will move all of this later
        def custom_collate_fn(batch):
            # check if ['text_chosen'] is in first batch element
            # Check if the first element of the batch is a dictionary
            if isinstance(batch[0]["text_chosen"][0], dict):
                return batch  # Return the batch as-is if it's a list of dicts
            else:
                return default_collate(batch)  # Use the default collate behavior otherwise

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=custom_collate_fn,  # if not args.pref_sets else None,
            shuffle=False,
            drop_last=False,
        )

        dataloader, model = accelerator.prepare(dataloader, reward_pipe.model)
        reward_pipe.model = model

        results = []
        scores_chosen = []
        scores_rejected = []
        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            logger.info(f"RM inference step {step}/{len(dataloader)}")

            if model_type == "Custom Classifier":
                print("Custom Classifier")
                text_rejected = [b["text_rejected"] for b in batch]
                text_chosen = [b["text_chosen"] for b in batch]
                # print(text_rejected)
                results_sub = reward_pipe(text_chosen, text_rejected, **reward_pipeline_kwargs)
                [results.append(1) if result else results.append(0) for result in results_sub.cpu().numpy().tolist()]
                scores_chosen.extend([None] * len(results_sub))
                scores_rejected.extend([None] * len(results_sub))
            else:
                print("other Classifier")
                rewards_chosen = reward_pipe(batch["text_chosen"], **reward_pipeline_kwargs)
                rewards_rejected = reward_pipe(batch["text_rejected"], **reward_pipeline_kwargs)

                # for each item in batch, record 1 if chosen > rejected
                # extra score from dict within batched results (e.g. logits)
                # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
                if isinstance(rewards_chosen[0], dict):
                    score_chosen_batch = [result["score"] for result in rewards_chosen]
                    score_rejected_batch = [result["score"] for result in rewards_rejected]
                # for classes that directly output scores (custom code)
                else:
                    print(rewards_chosen)
                    score_chosen_batch = rewards_chosen.cpu().numpy().tolist()
                    score_rejected_batch = rewards_rejected.cpu().numpy().tolist()

                # log results
                [
                    results.append(1) if chosen > rejected else results.append(0)
                    for chosen, rejected in zip(score_chosen_batch, score_rejected_batch)
                ]
                scores_chosen.extend(score_chosen_batch)
                scores_rejected.extend(score_rejected_batch)
        
    df = pd.read_csv(args.dataset)

    # 新增两列，列名为'chose'和'rejected'
    df['chosen_reward'] = scores_chosen
    df['rejected_reward'] = scores_rejected
    df['is_correct'] = results
    ACC = sum(results) / len(results)
    new_row = {'id': len(results), 'prompt': len(results), 'subset': len(results), 'chosen': len(results), 'rejected': len(results), 'chosen_reward': sum(results), 'rejected_reward': len(results), 'is_correct': ACC}
    df.loc[len(df)] = new_row

    # 保存为新文件
    df.to_csv(args.results, index=False)

if __name__ == "__main__":
    main()
