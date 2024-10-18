import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import json
from tqdm import tqdm
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# model_name = "Skywork-Reward-Gemma-2-27B"
model_name = "Skywork-Reward-Llama-3.1-8B"


model_path = "/home/jovyan/extra_storage/zey/models/" + model_name
# model_name = model_path.split('/')[-1]
rm = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(model_path)

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

dataset_names = [
    # "Advbench_suffix_5_n",
    # "Advbench_suffix_5_r",
    # "mixeval_freeform",
    # "mixeval_hard_freeform"
    # "Advbench_suffix_Yi5",
    # "Advbench_suffix_intern5"
    # 'mixeval_freeform_chat5',
    # 'mixeval_hard_freeform_chat5',
    # 'mixeval_freeform_Yi5',
    # 'mixeval_hard_freeform_Yi5'
    # 'mixeval_freeform_Intern5',
    # 'mixeval_hard_freeform_Intern5',
    # 'mixeval_freeform_multi_1',
    # 'mixeval_hard_freeform_multi_1'
    # 'Advbench_suffix_multi_1'
    # 'mixeval_freeform_multi_3',
    # 'mixeval_hard_freeform_multi_3',
    # 'mixeval_freeform_multi_2',
    # 'mixeval_hard_freeform_multi_2'
    # 'mixeval_freeform_DeepSeek_5',
    # 'mixeval_hard_freeform_DeepSeek_5'
    # 'mixeval_freeform_Llama-3.1-70B_5',
    # 'mixeval_hard_freeform_Llama-3.1-70B_5'
    # 'mixeval_freeform_Qwen2.5-72B_5',
    # 'mixeval_hard_freeform_Qwen2.5-72B_5',
    # 'arena-hard_DeepSeek5'
    # 'arena-hard_Qwen2.5-72B5'
    # 'arena-hard_Mistral-Large5'
    'mixeval_freeform_Mistral-Large_5',
    'mixeval_hard_freeform_Mistral-Large_5'
]
for dataset_name in dataset_names:
    input_path = '/home/jovyan/share_fudan/harmless/reward-bench-new/data_csv/' + dataset_name + '/' + dataset_name + '.csv'
    # input_path = "/home/jovyan/share_fudan/harmless/reward-bench-new/data_csv/Advbench_suffix_5/Advbench_suffix_5.csv"
    # input_path = "/home/jovyan/share_fudan/harmless/reward-bench-new/data_csv/ifeavl_5/ifeval_5.csv"
    df = pd.read_csv(input_path)
    data_dict = {col: df[col].tolist() for col in df.columns}
    scores_1 = []
    scores_2 = []
    scores_3 = []
    scores_4 = []
    scores_5 = []
    results = []
    for i in tqdm(range(len(data_dict["id"]))):
    # for i in [43137, 43138, 43139]:
        prompt = json.loads(data_dict["prompt"][i])
        # print(prompt)
        # print(data_dict["chosen"][i])
        model_1 = prompt + [{"role": "assistant", "content": str(data_dict["ans_1"][i])}]
        model_2 = prompt + [{"role": "assistant", "content": str(data_dict["ans_2"][i])}]
        model_3 = prompt + [{"role": "assistant", "content": str(data_dict["ans_3"][i])}]
        model_4 = prompt + [{"role": "assistant", "content": str(data_dict["ans_4"][i])}]
        model_5 = prompt + [{"role": "assistant", "content": str(data_dict["ans_5"][i])}]
        model_1_formatted = rm_tokenizer.apply_chat_template(model_1, tokenize=False)
        model_2_formatted = rm_tokenizer.apply_chat_template(model_2, tokenize=False)
        model_3_formatted = rm_tokenizer.apply_chat_template(model_3, tokenize=False)
        model_4_formatted = rm_tokenizer.apply_chat_template(model_4, tokenize=False)
        model_5_formatted = rm_tokenizer.apply_chat_template(model_5, tokenize=False)
        model_1_tokenized = rm_tokenizer(model_1_formatted, return_tensors="pt")
        model_2_tokenized = rm_tokenizer(model_2_formatted, return_tensors="pt")
        model_3_tokenized = rm_tokenizer(model_3_formatted, return_tensors="pt")
        model_4_tokenized = rm_tokenizer(model_4_formatted, return_tensors="pt")
        model_5_tokenized = rm_tokenizer(model_5_formatted, return_tensors="pt")
        with torch.no_grad():
            score1 = rm(**model_1_tokenized).logits[0][0].item()
            score2 = rm(**model_2_tokenized).logits[0][0].item()
            score3 = rm(**model_3_tokenized).logits[0][0].item()
            score4 = rm(**model_4_tokenized).logits[0][0].item()
            score5 = rm(**model_5_tokenized).logits[0][0].item()
        # print("score1: ", score1,  ", score2: ", score2)
        scores_1.append(score1)
        scores_2.append(score2)
        scores_3.append(score3)
        scores_4.append(score4)
        scores_5.append(score5)
        # results += select_best
        # if score1 > score2: results.append(1)
        # else: results.append(0)

    results = select_best(scores_1, scores_2, scores_3, scores_4, scores_5)
    # df = pd.read_csv(args.dataset)

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

    output_path = "/home/jovyan/share_fudan/harmless/reward-bench-new/result_bon_r/"+ dataset_name +"_" + model_name + ".csv"
    # 保存为新文件
    df.to_csv(output_path, index=False)
