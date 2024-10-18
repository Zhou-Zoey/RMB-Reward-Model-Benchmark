import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import json
from tqdm import tqdm
import os

# Load model and tokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# device = "cuda:1,2"
# model_name = "Skywork-Reward-Gemma-2-27B"
model_name = "Skywork-Reward-Llama-3.1-8B"
model_path = "/home/jovyan/extra_storage/zey/models/" + model_name
rm = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(model_path)

dataset_names = [
    "harmless_final_data_pair"
]
# dataset_name = '000current_prompt_fullaif_balanced_pairs_c2'
# dataset_name = '0905_reward_bench_consist'
for dataset_name in dataset_names:
    input_path = '/home/jovyan/share_fudan/harmless/reward-bench-new/data_csv/' + dataset_name + '/' + dataset_name + '.csv'
    df = pd.read_csv(input_path)
    data_dict = {col: df[col].tolist() for col in df.columns}
    scores_chosen = []
    scores_rejected = []
    results = []
    for i in tqdm(range(len(data_dict["id"]))):
        # try:
        prompt = json.loads(data_dict["prompt"][i])
        text_chosen = prompt + [{"role": "assistant", "content": str(data_dict["chosen"][i])}]
        text_rejected = prompt + [{"role": "assistant", "content": str(data_dict["rejected"][i])}]

        # Format and tokenize the conversations
        chosen_formatted = rm_tokenizer.apply_chat_template(text_chosen, tokenize=False)
        rejected_formatted = rm_tokenizer.apply_chat_template(text_rejected, tokenize=False)
        chosen_tokenized = rm_tokenizer(chosen_formatted, return_tensors="pt")
        rejected_tokenized = rm_tokenizer(rejected_formatted, return_tensors="pt")
        # Get the reward scores
        with torch.no_grad():
            score1 = rm(**chosen_tokenized).logits[0][0].item()
            score2 = rm(**rejected_tokenized).logits[0][0].item()
        # print("score1: ", score1,  ", score2: ", score2)
        scores_chosen.append(str(score1))
        scores_rejected.append(str(score2))
        if score1 > score2: results.append(1)
        else: results.append(0)
        # except Exception as e:
        #     print("ERROR: ", data_dict["id"])
        #     print(e)


    # df = pd.read_csv(args.dataset)

    # 新增两列，列名为'chose'和'rejected'
    df['chosen_reward'] = scores_chosen
    df['rejected_reward'] = scores_rejected
    df['is_correct'] = results
    ACC = sum(results) / len(results)
    new_row = {'id': len(results), 'prompt': len(results), 'subset': len(results), 'chosen': len(results), 'rejected': len(results), 'chosen_reward': sum(results), 'rejected_reward': len(results), 'is_correct': ACC}
    df.loc[len(df)] = new_row

    output_path = "/home/jovyan/share_fudan/harmless/reward-bench-new/result_balanced_c2/"+ dataset_name +"_" + model_name + ".csv"
    # 保存为新文件
    df.to_csv(output_path, index=False)


# prompt = "Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?"
# response1 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."
# response2 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among her 2 siblings (2 people in total). 9 ÷ 2 = 4.5 apples each. Each person gets 4 apples."

# conv1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}]
# conv2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response2}]

# # Format and tokenize the conversations
# conv1_formatted = rm_tokenizer.apply_chat_template(conv1, tokenize=False)
# conv2_formatted = rm_tokenizer.apply_chat_template(conv2, tokenize=False)
# conv1_tokenized = rm_tokenizer(conv1_formatted, return_tensors="pt").to(device)
# conv2_tokenized = rm_tokenizer(conv2_formatted, return_tensors="pt").to(device)

# # Get the reward scores
# with torch.no_grad():
#     score1 = rm(**conv1_tokenized).logits[0][0].item()
#     score2 = rm(**conv2_tokenized).logits[0][0].item()
# print(f"Score for response 1: {score1}")
# print(f"Score for response 2: {score2}")

# Output:
# Score for response 1: 9.1875
# Score for response 2: -17.875




##################################Skywork/Skywork-Reward-Llama-3.1-8B


# import torch

# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# # Load model and tokenizer
# device = "cuda:0"
# model_name = "Skywork/Skywork-Reward-Llama-3.1-8B"
# rm = AutoModelForSequenceClassification.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     device_map=device,
#     attn_implementation="flash_attention_2",
#     num_labels=1,
# )
# rm_tokenizer = AutoTokenizer.from_pretrained(model_name)

# prompt = "Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?"
# response1 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."
# response2 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among her 2 siblings (2 people in total). 9 ÷ 2 = 4.5 apples each. Each person gets 4 apples."

# conv1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}]
# conv2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response2}]

# # Format and tokenize the conversations
# conv1_formatted = rm_tokenizer.apply_chat_template(conv1, tokenize=False)
# conv2_formatted = rm_tokenizer.apply_chat_template(conv2, tokenize=False)
# conv1_tokenized = rm_tokenizer(conv1_formatted, return_tensors="pt").to(device)
# conv2_tokenized = rm_tokenizer(conv2_formatted, return_tensors="pt").to(device)

# # Get the reward scores
# with torch.no_grad():
#     score1 = rm(**conv1_tokenized).logits[0][0].item()
#     score2 = rm(**conv2_tokenized).logits[0][0].item()
# print(f"Score for response 1: {score1}")
# print(f"Score for response 2: {score2}")

# # Output:
# # Score for response 1: 12.625
# # Score for response 2: -15.25


