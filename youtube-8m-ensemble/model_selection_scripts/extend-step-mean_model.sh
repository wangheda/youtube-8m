#!/bin/bash

candidates_conf="$1"

train_path=/Youtube-8M/model_predictions/ensemble_train
model_path="../model/mean_model"
all_models_conf="${model_path}/all_models.conf"

for candidates in $(cat $candidates_conf); do
  echo "$candidates"
  train_data_patterns=$(python model_selection_scripts/get_patterns.py --train_path="$train_path" --candidates="$candidates")
  CUDA_VISIBLE_DEVICES=1 python eval.py \
      --model_checkpoint_path="${model_path}/model.ckpt-0" \
      --train_dir="${model_path}" \
      --model="MeanModel" \
      --echo_gap=True \
      --eval_data_patterns="$train_data_patterns" | tail -n 1
done
