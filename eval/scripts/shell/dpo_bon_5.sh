#!/bin/bash
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

export CUDA_VISIBLE_DEVICES="0"

# models=("mistral0.1" "llama2-7b" "vicuna" "mixtral" "llama2-70b")
models=(
    # 'ArmoRM-Llama3-8B-v0.1'
    # 'Eurus-RM-7b'
    'stablelm-2-12b-chat'
    # 'Starling-RM-34B'
    # 'zephyr-7b-alpha'
    # 'UltraRM-13b'
    # 'PairRM-hf'

)
# datasets=("9unclear180-3000tokens_Closed QA8259-500" "37clear316-3000tokens_Closed QA8259-500" "46-tag1-413-clear level: 1-436-Closed QA566" "60clear255-3000tokens_Brainstorming333-333" "67-tag1-1943-clear level: 1-2084-Generation3088")
datasets=(
    # 'harmless_final_data'
    # 'helpful_final_data'
    # 'harmless_final_data_pair'
    # 'helpful_final_data_pair'
    # '0718_results_pairs_qwen_final'
    # '0718_results_pairs_qwen_finalunfiltered'
    # '000current_prompt_fullaif_balanced_pairs'
    'ifeval_5'
)


# --tokenizer "01-ai/Yi-34B-Chat" \
# --chat_template=openbmb \
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "$dataset"
        echo "$model"
        python my_run_dpo_bon.py \
        --model="/home/jovyan/share_fudan/harmless/models/${model}" \
        --dataset_dir="/home/jovyan/share_fudan/harmless/reward-bench-new/data_csv/${dataset}" \
        --dataset="/home/jovyan/share_fudan/harmless/reward-bench-new/data_csv/${dataset}/${dataset}.csv" \
        --results="/home/jovyan/share_fudan/harmless/reward-bench-new/result_bon/${dataset}_${model}_res.csv" \
        > "log_csv/${dataset}_${model}".log 2>&1 &
    done
done
