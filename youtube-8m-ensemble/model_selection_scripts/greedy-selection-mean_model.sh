#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
model_name=$1

extend_step_script="${DIR}/extend-step-mean_model.sh"
train_path=/Youtube-8M/model_predictions/ensemble_train
model_path="${DIR}/../../model/${model_name}"
all_models_conf="${model_path}/all_models.conf"

if [ -f $all_models_conf ]; then 

  top_models_conf="${all_models_conf}"
  len_k_models_conf="${all_models_conf}"

  for step in {1..19}; do 

      if [ $step -gt 1 ]; then
          len_k_models_conf="${model_path}/len_${step}_models.conf"
          python ${DIR}/get_extend_candidates.py --top_k_file="$top_models_conf" --all_models_conf="${all_models_conf}" > $len_k_models_conf
      fi

      bash $extend_step_script $model_name ${len_k_models_conf} > ${model_path}/len_${step}_models.log
      python ${DIR}/get_top_k.py --top_k=3 --log_file="${model_path}/len_${step}_models.log" --sorted_log_file="${model_path}/len_${step}_models.sorted.log" > ${model_path}/top_${step}_models.conf

      top_models_conf="${model_path}/top_${step}_models.conf"
  done

else

  echo $all_models_conf not found, did nothing

fi
