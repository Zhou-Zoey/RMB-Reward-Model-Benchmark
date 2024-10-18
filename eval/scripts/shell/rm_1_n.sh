export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

export CUDA_VISIBLE_DEVICES="3"
# models=("mistral0.1" "llama2-7b" "vicuna" "mixtral" "llama2-70b")
models=(
    # 'ArmoRM-Llama3-8B-v0.1'
    # 'Eurus-RM-7b'
    # 'stablelm-2-12b-chat'
    # 'Starling-RM-34B'
    # 'internlm2-7b-reward'
    # 'internlm2-20b-reward'
    'tulu-v2.5-13b-preference-mix-rm'
    # 'Llama3-70B-SteerLM-RM'
    # 'zephyr-7b-alpha'
    # 'UltraRM-13b'
    # 'PairRM-hf'

)

datasets=(
    # 'harmless_final_data'
    # 'helpful_final_data'
    # 'harmless_final_data_pair'
    # 'helpful_final_data_pair'
    # '0718_results_pairs_qwen_final'
    # '0718_results_pairs_qwen_finalunfiltered'
    # '000current_prompt_fullaif_balanced_pairs'
    '0905_reward_bench_consist'
)

# --tokenizer "01-ai/Yi-34B-Chat" \
# --chat_template=openbmb \
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "$dataset"
        echo "$model"
        python my_run_rm.py \
        --model="/home/jovyan/share_fudan/harmless/models/${model}" \
        --dataset_dir="/home/jovyan/share_fudan/harmless/reward-bench-new/data_csv/${dataset}" \
        --dataset="/home/jovyan/share_fudan/harmless/reward-bench-new/data_csv/${dataset}/${dataset}.csv" \
        --results="/home/jovyan/share_fudan/harmless/reward-bench-new/result_csv/${dataset}_${model}_res.csv" \
        > "log_csv/${dataset}_${model}".log 2>&1 &
    done
done
