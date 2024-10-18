export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

export CUDA_VISIBLE_DEVICES="1,2"

# python sky_work.py > "log_csv_2/000current_prompt_fullaif_balanced_pairs_c2_Skywork-Reward-Llama-3.1-8B".log 2>&1 &
python sky_work.py > "log_csv_2/000current_prompt_fullaif_balanced_pairs_c2_Skywork-Reward-Gemma-2-27B".log 2>&1 &