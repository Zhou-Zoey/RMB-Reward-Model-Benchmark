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
from trl.trainer.utils import DPODataCollatorWithPadding

from rewardbench import (
    DPO_MODEL_CONFIG,
    save_to_hub,
)
from bon_utils.dpo import DPOInference
from bon_utils.utils import torch_dtype_mapping, load_eval_dataset
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument("--ref_model", type=str, default=None, help="path to model")
    parser.add_argument(
        "--ref_free_type", type=str, default="avg", help="type of reference free normalization (norm, avg, or sum)"
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer")
    parser.add_argument("--dataset_dir", type=str, required=True, help="path to data_dir")
    parser.add_argument("--dataset", type=str, required=True, help="path to data")
    parser.add_argument("--results", type=str, required=True, help="path to results")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=6, help="batch size for inference")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=True, help="directly load model instead of pipeline"
    )
    parser.add_argument("--debug", action="store_true", default=False, help="use only 10 examples")
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    parser.add_argument(
        "--not_quantized", action="store_true", help="disable quantization for models that are quantized by default"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32", "float64"],
        help="PyTorch dtype (default: float16)",
    )
    args = parser.parse_args()
    args.torch_dtype = torch_dtype_mapping(args.torch_dtype)
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

def select_best(scores_1, scores_2, scores_3, scores_4, scores_5):
    # scores_list = {
    #     "1": scores_1,
    #     "2": scores_2,
    #     "3": scores_3,
    #     "4": scores_4,
    #     "5": scores_5
    # }
    results = []
    for s1,s2,s3,s4,s5 in zip(scores_1, scores_2, scores_3, scores_4, scores_5):
        scores = [s1,s2,s3,s4,s5]
        # max_score = max(scores)
        max_score = scores[0]
        k = 0
        for i in range(len(scores)):
            if scores[i] > max_score:
                max_score = scores[i]
                k = i
        results.append(str(k+1))
        # results.append([n for n, score_list in scores_list.items() if max_score in score_list and score_list[scores.index(max_score)] == max_score][0])

    return results


def main():
    args = get_args()
    accelerator = Accelerator()

    ###############
    # Setup logging
    ###############
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
        '/home/jovyan/share_fudan/harmless/models/zephyr-7b-alpha':'HuggingFaceH4/zephyr-7b-alpha',
        '/home/jovyan/share_fudan/harmless/models/UltraRM-13b':'openbmb/UltraRM-13b',
        '/home/jovyan/share_fudan/harmless/models/PairRM-hf':'llm-blender/PairRM-hf',
        '/home/jovyan/share_fudan/harmless/models/internlm2-20b-reward': 'internlm/internlm2-20b-reward',
        '/home/jovyan/share_fudan/harmless/models/internlm2-7b-reward': 'internlm/internlm2-7b-reward',
        '/home/jovyan/share_fudan/harmless/models/Mixtral-8x7B-Instruct-v0.1': 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    }

    model_config = load_config('/home/jovyan/share_fudan/harmless/reward-bench-new/scripts/configs/eval_configs.yaml')
    model_name = p2n[args.model]
    config_dict = get_parameters(model_config, model_name)

    if args.model in DPO_MODEL_CONFIG:
        config = DPO_MODEL_CONFIG[args.model]
    else:
        config = DPO_MODEL_CONFIG["default"]
    ref_model_name = config_dict['ref_model']
    logger.info(f"Using dpo model config: {config}")

    model_builder = config["model_builder"]
    tokenizer_builder = config["tokenizer_builder"]

    # check datatype from argparse
    if args.torch_dtype == torch.bfloat16:
        logger.warning("Loading weights directly as bfloat16 for PyTorch dtype")
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    assert args.model != args.ref_model, "policy and reference model should be different"
    # load chat template
    # chat_template = args.chat_template
    # conv = get_conv_template(chat_template)
    if 'chat_template' in config_dict.keys():
        chat_template = config_dict['chat_template']
    else:
        chat_template = "tulu"
    try:
        conv = get_conv_template(chat_template)
    except Exception as e:
        conv = get_conv_template("tulu")

    # define reference free
    # if args.ref_model is None:
    #     ref_free = True
    #     logger.info("Running reference free DPO - no reference model provided")
    # else:
    #     ref_free = False
    #     logger.info(f"Running DPO with reference model {args.ref_model}")
    if ref_model_name == '' or ref_model_name is None:
        ref_free = True
        logger.info("Running reference free DPO - no reference model provided")
    else:
        ref_free = False
        logger.info(f"Running DPO with reference model {ref_model_name}")

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    # tokenizer_path = args.tokenizer if args.tokenizer else args.model
    if model_name != config_dict['tokenizer']:
        tokenizer_path = config_dict['tokenizer']
    else:
        tokenizer_path = args.model
    tokenizer = tokenizer_builder(tokenizer_path, trust_remote_code=args.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    # if no BOS token, set as pad token, e.g. QWEN models
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset, subsets = load_eval_dataset(
        core_set=False,
        EXTRA_PREF_SETS = "/home/jovyan/share_fudan/harmless/reward-bench-new/data_csv/ifeavl_5",
        conv=conv,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["model_1", "model_2", "model_3", "model_4", "model_5", "id", "prompt"],
    )

    dataset = dataset.remove_columns("id")
    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(10))
        subsets = subsets[:10]

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size

    if (
        ("llama-3" in args.model)
        or ("Llama3" in args.model)
        or ("Llama-3" in args.model)
        or ("LLaMA3" in args.model)
        or args.not_quantized
    ):
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }
        model_kwargs_ref = {
            "device_map": "auto",
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {
            "load_in_8bit": False,
            "device_map": "auto",
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }
        model_kwargs_ref = {
            "load_in_8bit": False,
            "device_map": "auto",
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }

    model = model_builder(
        args.model,
        trust_remote_code=args.trust_remote_code,
        **model_kwargs,
    )

    if ref_free:
        ref_model = None
    else:
        ref_model = model_builder(
            ref_model_name,
            trust_remote_code=args.trust_remote_code,
            **model_kwargs_ref,
        )

    # use internal inference functions in DPO trainer
    dpo = DPOInference(
        model,
        ref_model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        ref_free_norm=args.ref_free_type,
        # norm is norm, avg is average, sum is sum
    )
    # tokenize dataset
    column_names = list(dataset.features)

    tokenized_dataset = dataset.map(dpo.tokenize_row, remove_columns=column_names)

    BATCH_SIZE = config_dict['batch_size']
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=DPODataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=dpo.label_pad_token_id,
            is_encoder_decoder=dpo.is_encoder_decoder,
        ),
        # collate_fn = lambda x: x, # fix weird batching error
        shuffle=False,
        drop_last=False,
    )
    results = []
    # scores_chosen = []
    # scores_rejected = []
    scores_1 = []
    scores_2 = []
    scores_3 = []
    scores_4 = []
    scores_5 = []

    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        logger.info(f"RM inference step {step}/{len(dataloader)}")

        rewards_1, rewards_2, rewards_3, rewards_4, rewards_5 = dpo.inference_step(batch, ref_free=ref_free)

        # for each item in batch, record 1 if chosen > rejected
        # extra score from dict within batched results (e.g. logits)
        # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        if isinstance(rewards_1[0], dict):
            scores_1_batch = [result["score"] for result in rewards_1]
            scores_2_batch = [result["score"] for result in rewards_2]
            scores_3_batch = [result["score"] for result in rewards_3]
            scores_4_batch = [result["score"] for result in rewards_4]
            scores_5_batch = [result["score"] for result in rewards_5]
            # scores_rejected_batch = [result["score"] for result in rewards_rejected]
        # for classes that directly output scores (custom code)
        else:
            scores_1_batch = rewards_1.float().cpu().numpy().tolist()  # convert to float for bfloat16 case
            scores_2_batch = rewards_2.float().cpu().numpy().tolist()
            scores_3_batch = rewards_3.float().cpu().numpy().tolist()
            scores_4_batch = rewards_4.float().cpu().numpy().tolist()
            scores_5_batch = rewards_5.float().cpu().numpy().tolist()
            # scores_rejected_batch = rewards_rejected.float().cpu().numpy().tolist()

        # [
        #     results.append(1) if chosen > rejected else results.append(0)
        #     for chosen, rejected in zip(scores_chosen_batch, scores_rejected_batch)
        # ]
        results += select_best(scores_1_batch, scores_2_batch, scores_3_batch, scores_4_batch, scores_5_batch)
        scores_1 += scores_1_batch
        scores_2 += scores_2_batch
        scores_3 += scores_3_batch
        scores_4 += scores_4_batch
        scores_5 += scores_5_batch
        # scores_rejected += scores_rejected_batch

    df = pd.read_csv("/home/jovyan/share_fudan/harmless/reward-bench-new/data_csv/ifeavl_5/ifeavl_5.csv")

    # 新增两列，列名为'chose'和'rejected'
    df['reward_1'] = scores_1
    df['reward_2'] = scores_2
    df['reward_3'] = scores_3
    df['reward_4'] = scores_4
    df['reward_5'] = scores_5
    df['best_ans'] = results
    # ACC = sum(results) / len(results)
    # new_row = {'id': len(results), 'prompt': len(results), 'subset': len(results), 'chosen': len(results), 'rejected': len(results), 'chosen_reward': sum(results), 'rejected_reward': len(results), 'is_correct': ACC}
    # df.loc[len(df)] = new_row

    # 保存为新文件
    df.to_csv(args.results, index=False)

    # ############################
    # # Print & process results
    # ############################
    # # add column for results for easy printing
    # out_dataset = dataset.add_column("results", results)

    # # add subsets back (removed so it's not handled by cuda)
    # out_dataset = out_dataset.add_column("subset", subsets)
    # # add scores_chosen and scores_rejected to the dataset
    # out_dataset = out_dataset.add_column("scores_chosen", scores_chosen)
    # out_dataset = out_dataset.add_column("scores_rejected", scores_rejected)

    # results_grouped = {}
    # results_grouped["model"] = args.model
    # results_grouped["ref_model"] = args.ref_model
    # results_grouped["model_type"] = "DPO"  # TODO add options for references free, DPO-ref-free, or DPO-normalized
    # if ref_free:
    #     results_grouped["model_type"] = "DPO Ref. Free"
    #     save_modifier = "_ref_free"
    # else:
    #     save_modifier = ""
    # results_grouped["chat_template"] = args.chat_template if not hasattr(tokenizer, "chat_template") else "tokenizer"
    # # print per subset and log into results_grouped file
    # present_subsets = np.unique(subsets)
    # for subset in present_subsets:
    #     subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
    #     num_correct = sum(subset_dataset["results"])
    #     num_total = len(subset_dataset["results"])
    #     print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
    #     results_grouped[subset] = num_correct / num_total

    # # log leaderboard aggregated results
    # if not args.pref_sets:
    #     results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
    #     print(results_leaderboard)

    # ############################
    # # Upload results to hub
    # ############################
    # sub_path = "eval-set/" if not args.pref_sets else "pref-sets/"
    # results_url = save_to_hub(
    #     results_grouped,
    #     args.model + save_modifier,
    #     sub_path,
    #     args.debug,
    #     local_only=args.do_not_save,
    #     save_metrics_for_beaker=not args.disable_beaker_save,
    # )
    # if not args.do_not_save:
    #     logger.info(f"Uploaded reward model results to {results_url}")

    # # upload chosen-rejected with scores
    # # create new json with scores and upload
    # scores_dict = out_dataset.to_dict()
    # scores_dict["model"] = args.model
    # scores_dict["model_type"] = "DPO"
    # scores_dict["chat_template"] = args.chat_template
    # sub_path_scores = "eval-set-scores/" if not args.pref_sets else "pref-sets-scores/"

    # scores_url = save_to_hub(
    #     scores_dict, args.model + save_modifier, sub_path_scores, args.debug, local_only=args.do_not_save
    # )
    # logger.info(f"Uploading chosen-rejected text with scores to {scores_url}")


if __name__ == "__main__":
    main()
