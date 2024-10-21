export CUDA_VISIBLE_DEVICES="4"

model_path=''
dataset_path=''
result_path=''

models=(
    # 'ArmoRM-Llama3-8B-v0.1'
    # 'Eurus-RM-7b'
    'stablelm-2-12b-chat'
    # 'Starling-RM-34B'
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


for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "$dataset"
        echo "$model"
        python my_run_dpo.py \
        --model="${model_path}/${model}" \
        --dataset_dir="${dataset_path}/${dataset}" \
        --dataset="${dataset_path}/${dataset}/${dataset}.csv" \
        --results="${result_path}/${dataset}_${model}_res.csv" \
        > "log/${dataset}_${model}".log 2>&1 &
    done
done
