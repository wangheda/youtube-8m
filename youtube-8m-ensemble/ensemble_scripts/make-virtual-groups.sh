#!/bin/bash

group_dir="../model/virtual_grouping"

for file in $(ls $group_dir); do
  if [[ $file =~ ^virtual_group ]] && [[ $file =~ conf$ ]]; then
    group_name="${file:0:-5}"
    model="virtual_grouping/${group_name}"
    model_dir="${group_dir}/${group_name}"
    echo "training $model ..."

    if [ ! -d $model_dir ]; then
      conf="${group_dir}/${group_name}.conf"
      bash ensemble_scripts/train-matrix_model.sh $model $conf
      bash ensemble_scripts/eval-matrix_model.sh $model $conf
    fi

    for part in ensemble_train ensemble_validate test; do
      output_dir="/Youtube-8M/model_predictions/${part}/${model}"
      if [ ! -d $output_dir ]; then
        bash ensemble_scripts/auto-preensemble-matrix_model.sh $model $conf $part
      fi
    done
  fi
done
