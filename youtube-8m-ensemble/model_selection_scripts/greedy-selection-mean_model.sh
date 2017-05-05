#!/bin/bash

extend_step_script="model_selection_scripts/extend-step-mean_model.sh"
train_path=/Youtube-8M/model_predictions/ensemble_train
model_path="../model/mean_model"
all_models_conf="${model_path}/all_models.conf"

top_models_conf="${all_models_conf}"
len_k_models_conf="${all_models_conf}"

for step in {1..19}; do 

    if [ $step -gt 1 ]; then
        len_k_models_conf="${model_path}/len_${step}_models.conf"
        python model_selection_scripts/get_extend_candidates.py --top_k_file="$top_models_conf" --all_models_conf="${all_models_conf}" > $len_k_models_conf
    fi

    bash $extend_step_script ${len_k_models_conf} > ${model_path}/len_${step}_models.log
    python model_selection_scripts/get_top_k.py --log_file="${model_path}/len_${step}_models.log" --sorted_log_file="${model_path}/len_${step}_models.sorted.log" > ${model_path}/top_${step}_models.conf

    top_models_conf="${model_path}/top_${step}_models.conf"
done
