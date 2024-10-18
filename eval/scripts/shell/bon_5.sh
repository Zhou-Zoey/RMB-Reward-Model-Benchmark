export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

# models=("mistral0.1" "llama2-7b" "vicuna" "mixtral" "llama2-70b")
export CUDA_VISIBLE_DEVICES="4"

models=(
    # 'ArmoRM-Llama3-8B-v0.1'
    # 'Eurus-RM-7b'
    # 'tulu-v2.5-13b-preference-mix-rm'
    'Starling-RM-34B'
)

datasets=(
    # 'mixeval_freeform_Qwen2.5-72B_5'
    # 'mixeval_hard_freeform_Qwen2.5-72B_5'
    # 'arena-hard_DeepSeek5'
    # 'arena-hard_Qwen2.5-72B5'
    # 'arena-hard_Mistral-Large5'
    'mixeval_freeform_Mistral-Large_5'
    # 'mixeval_hard_freeform_Mistral-Large_5'
)

# export CUDA_VISIBLE_DEVICES="0"
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "$dataset"
        echo "$model"
        python my_run_bon.py \
        --model="/home/jovyan/share_fudan/harmless/models/${model}" \
        --key=${dataset} \
        --dataset_dir="/home/jovyan/share_fudan/harmless/reward-bench-new/data_csv/${dataset}" \
        --dataset="/home/jovyan/share_fudan/harmless/reward-bench-new/data_csv/${dataset}/${dataset}.csv" \
        --results="/home/jovyan/share_fudan/harmless/reward-bench-new/result_bon_r/${dataset}_${model}_res.csv" \
        > "log_csv_multi/${dataset}_${model}".log 2>&1 &
        wait
    done
done