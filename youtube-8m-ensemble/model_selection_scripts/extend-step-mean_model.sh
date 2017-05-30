#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
model_name="$1"
candidates_conf="$2"

train_path=/Youtube-8M/model_predictions_for_selection/ensemble_train
model_path="${DIR}/../../model/${model_name}"
all_models_conf="${model_path}/all_models.conf"

for candidates in $(cat $candidates_conf); do
  echo "$candidates"
  train_data_patterns=$(python ${DIR}/get_patterns.py --train_path="$train_path" --candidates="$candidates")
  CUDA_VISIBLE_DEVICES=1 python ${DIR}/../eval.py \
      --model_checkpoint_path="${model_path}/model.ckpt-0" \
      --train_dir="${model_path}" \
      --model="MeanModel" \
      --echo_gap=True \
      --batch_size=1024 \
      --eval_data_patterns="$train_data_patterns" | tail -n 1
done
