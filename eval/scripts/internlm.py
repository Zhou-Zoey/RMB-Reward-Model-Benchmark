import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import json
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# model_name = "internlm2-7b-reward"
model_name = "internlm2-20b-reward"
# dataset_name = 'harmless_final_data_pair'
# dataset_name = 'helpful_final_data_pair'
dataset_name = '0905_reward_bench_consist'
# dataset_name = '0718_results_pairs_qwen_finalunfiltered'

model_path = '/home/jovyan/share_fudan/harmless/models/' +  model_name
# model_name = model_path.split('/')[-1]
model = AutoModel.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

input_path = '/home/jovyan/share_fudan/harmless/reward-bench-new/data_csv/' + dataset_name + '/' + dataset_name + '.csv'
df = pd.read_csv(input_path)
data_dict = {col: df[col].tolist() for col in df.columns}
scores_chosen = []
scores_rejected = []
results = []
for i in tqdm(range(len(data_dict["id"]))):
# for i in [43137, 43138, 43139]:
    try:
        prompt = json.loads(data_dict["prompt"][i])
        # print(prompt)
        # print(data_dict["chosen"][i])
        text_chosen = prompt + [{"role": "assistant", "content": str(data_dict["chosen"][i])}]
        text_rejected = prompt + [{"role": "assistant", "content": str(data_dict["rejected"][i])}]
        score1 = model.get_score(tokenizer, text_chosen)
        score2 = model.get_score(tokenizer, text_rejected)
        # print("score1: ", score1,  ", score2: ", score2)
        scores_chosen.append(str(score1))
        scores_rejected.append(str(score2))
        if score1 > score2: results.append(1)
        else: results.append(0)
    except Exception as e:
        print("ERROR: ", data_dict["id"])
        print(e)


# df = pd.read_csv(args.dataset)

# 新增两列，列名为'chose'和'rejected'
df['chosen_reward'] = scores_chosen
df['rejected_reward'] = scores_rejected
df['is_correct'] = results
ACC = sum(results) / len(results)
new_row = {'id': len(results), 'prompt': len(results), 'subset': len(results), 'chosen': len(results), 'rejected': len(results), 'chosen_reward': sum(results), 'rejected_reward': len(results), 'is_correct': ACC}
df.loc[len(df)] = new_row

output_path = "/home/jovyan/share_fudan/harmless/reward-bench-new/result_csv/"+ dataset_name +"_" + model_name + ".csv"
# 保存为新文件
df.to_csv(output_path, index=False)
# chat_1 = [
#     {"role": "user", "content": "Hello! What's your name?"},
#     {"role": "assistant", "content": "My name is InternLM2! A helpful AI assistant. What can I do for you?"}
# ]
# chat_2 = [
#     {"role": "user", "content": "Hello! What's your name?"},
#     {"role": "assistant", "content": "I have no idea."}
# ]


# # get reward score for a single chat
# score1 = model.get_score(tokenizer, chat_1)
# score2 = model.get_score(tokenizer, chat_2)
# print("score1: ", score1)
# print("score2: ", score2)
# # >>> score1:  0.767578125
# # >>> score2:  -2.22265625


# # batch inference, get multiple scores at once
# scores = model.get_scores(tokenizer, [chat_1, chat_2])
# print("scores: ", scores)
# # >>> scores:  [0.767578125, -2.22265625]


# # compare whether chat_1 is better than chat_2
# compare_res = model.compare(tokenizer, chat_1, chat_2)
# print("compare_res: ", compare_res)
# # >>> compare_res:  True


# # rank multiple chats, it will return the ranking index of each chat
# # the chat with the highest score will have ranking index as 0
# rank_res = model.rank(tokenizer, [chat_1, chat_2])
# print("rank_res: ", rank_res)  # lower index means higher score
# # >>> rank_res:  [0, 1]



##################################################20b
##https://github.com/NVIDIA/NeMo-Aligner
##https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/steerlm.html


# import torch
# from transformers import AutoModel, AutoTokenizer

# model = AutoModel.from_pretrained(
#     "internlm/internlm2-20b-reward",
#     device_map="cuda",
#     torch_dtype=torch.float16,
#     trust_remote_code=True,
# )
# tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-20b-reward", trust_remote_code=True)

# chat_1 = [
#     {"role": "user", "content": "Hello! What's your name?"},
#     {"role": "assistant", "content": "My name is InternLM2! A helpful AI assistant. What can I do for you?"}
# ]
# chat_2 = [
#     {"role": "user", "content": "Hello! What's your name?"},
#     {"role": "assistant", "content": "I have no idea."}
# ]


# # get reward score for a single chat
# score1 = model.get_score(tokenizer, chat_1)
# score2 = model.get_score(tokenizer, chat_2)
# print("score1: ", score1)
# print("score2: ", score2)
# # >>> score1:  0.767578125
# # >>> score2:  -2.22265625


# # batch inference, get multiple scores at once
# scores = model.get_scores(tokenizer, [chat_1, chat_2])
# print("scores: ", scores)
# # >>> scores:  [0.767578125, -2.22265625]


# # compare whether chat_1 is better than chat_2
# compare_res = model.compare(tokenizer, chat_1, chat_2)
# print("compare_res: ", compare_res)
# # >>> compare_res:  True


# # rank multiple chats, it will return the ranking index of each chat
# # the chat with the highest score will have ranking index as 0
# rank_res = model.rank(tokenizer, [chat_1, chat_2])
# print("rank_res: ", rank_res)  # lower index means higher score
# # >>> rank_res:  [0, 1]