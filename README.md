# RMB: Comprehensively Benchmarking Reward Models in LLM Alignment
*RMB* is a comprehensive RM benchmark that covers over 49 real-world scenarios and includes both *pairwise* and *Best-of-N (BoN)* evaluations to better reflect the effectiveness of RMs in *guiding alignment optimization*. 
## News and Updates
- **[11/10/2024]** Our report is on [**Arxiv**](https://arxiv.org/abs/2410.09893)!
## Overview
![Statistics of queries, pairwise set, Best-of-N test set in different scenarios under harmlessness goal](fig/harmless_dataset_table.pdf)
![Statistics of queries, pairwise set, Best-of-N test set in different scenarios under helpfulness goal](fig/helpful_dataset_table.pdf)
![Subcategories of helpfulness scenarios](fig/helpful_dataset_2.pdf)
## How to use

### Install

1. Clone this repository and navigate to RMB-Reward-Model-Benchmark folder
```bash
git clone https://github.com/Zhou-Zoey/RMB-Reward-Model-Benchmark.git
cd RMB-Reward-Model-Benchmark
```

2. Install Package

```Shell
conda create -n RMB python=3.10 -y
conda activate RMB
pip install --upgrade pip
pip install -r requirements.txt
```

3. Install additional packages for some Reward Model
```
pip install flash-attn --no-build-isolation
```

### Quick Usage

Eval Reward Model on our dataset

```Shell
cd eval/scripts/shell
bash  run_rm.sh
```

run_rm.sh
```bash
# your models root dir
# If you want download model from huggingface directly, no need to fill model_path
model_path=''
models=(
    'ArmoRM-Llama3-8B-v0.1'
    # 'Eurus-RM-7b'
    # 'stablelm-2-12b-chat'
    # 'Starling-RM-34B'
    # 'internlm2-7b-reward'
    # 'internlm2-20b-reward'
    # 'tulu-v2.5-13b-preference-mix-rm'
)

# your RMB_dataset path
dataset_path=''
datasets=(
    'Pairwise_set/Helpfulness/Brainstorming/Idea Development.json'
)

# your also can run the data folder
'''
datasets=(
    'Pairwise_set/Helpfulness/Brainstorming'
    'Pairwise_set/Helpfulness/Chat'
)
'''


# your results path
result_path='../RMB-Reward-Model-Benchmark/eval/results'

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "$dataset"
        echo "$model"
        python RMB-Reward-Model-Benchmark/eval/scripts/my_run_rm.py \
        --model_dir="${model_path}" \
        --model="${model_path}/${model}" \
        --dataset_dir="${dataset_path}/${dataset}" \
        --single_data=True \
        --dataset="${dataset_path}/${dataset}/${dataset}.json" \
        --results="${result_path}/${model}_result.json"
    done
done


# > "log/eval".log 2>&1 &
```


coming soon
## RMB Benchmark
The datasets we used to benchmark the reward models have been uploaded in the `/RMB_dataset` directory. We acknowledge that you might have different opinions on some of the annotations in the data, which is actually normal for preference data.

Note: There may be texts that are offensive in nature.
