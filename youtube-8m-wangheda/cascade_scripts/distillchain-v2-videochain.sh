#!/bin/bash

task=$1

model_name="videochain"
model_path="../model/${model_name}"

if [ $task == "train" ]; then
  predictions_data_pattern='/Youtube-8M/model_predictions_local/train/distillation/ensemble_v2_matrix_model/*.tfrecord'

  for i in {1..5}; do
    sub_model_name="distillchain_v2_video_dcc_$i"
    sub_model_path="${model_path}/${sub_model_name}"
    if [ ! -d $sub_model_path ]; then 
      mkdir -p $sub_model_path
      echo predictions = "$predictions_data_pattern"

      CUDA_VISIBLE_DEVICES=0 python train-with-predictions.py \
        --train_dir="$sub_model_path" \
        --train_data_pattern="/Youtube-8M/data/video/train/*.tfrecord" \
        --predictions_data_pattern="$predictions_data_pattern" \
        --distillation_features=False \
        --distillation_as_input=True \
        --frame_features=False \
        --feature_names="mean_rgb,mean_audio" \
        --feature_sizes="1024,128" \
        --model=DistillchainDeepCombineChainModel \
        --moe_num_mixtures=4 \
        --deep_chain_layers=3 \
        --deep_chain_relu_cells=256 \
        --data_augmenter=NoiseAugmenter \
        --input_noise_level=0.1 \
        --multitask=True \
        --label_loss=MultiTaskCrossEntropyLoss \
        --support_type="label,label,label" \
        --support_loss_percent=0.05 \
        --base_learning_rate=0.01 \
        --keep_checkpoint_every_n_hour=5.0 \
        --keep_checkpoint_interval=6 \
        --num_epochs=2 \
        --batch_size=1024
    fi

    if [ "$i" -ne "5" ]; then 
      # inference model 1
      infer_data_path="/Youtube-8M/model_predictions_local/train/${model_name}/${sub_model_name}"
      if [ ! -d $infer_data_path ]; then
        echo predictions = "$predictions_data_pattern"

        CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble-with-predictions.py \
          --output_dir="$infer_data_path" \
          --train_dir="$sub_model_path" \
          --input_data_pattern="/Youtube-8M/data/video/train/*.tfrecord" \
          --distill_data_pattern="$predictions_data_pattern" \
          --frame_features=False \
          --feature_names="mean_rgb,mean_audio" \
          --feature_sizes="1024,128" \
          --distillation_features=False \
          --distillation_as_input=True \
          --model=DistillchainDeepCombineChainModel \
          --moe_num_mixtures=4 \
          --deep_chain_layers=3 \
          --deep_chain_relu_cells=256 \
          --batch_size=1024 \
          --file_size=4096
      fi
    fi

    # next predictions pattern
    predictions_data_pattern="${predictions_data_pattern},${infer_data_path}/*.tfrecord"

    if [ -d $sub_model_path ]; then 
      for part in ensemble_train ensemble_validate test; do
        infer_data_path="/Youtube-8M/model_predictions_local/${part}/${model_name}/${sub_model_name}"
        if [ ! -d $infer_data_path ]; then
          preensemble_data_pattern="/Youtube-8M/model_predictions_local/${part}/distillation/ensemble_v2_matrix_model/*.tfrecord"
          for j in {1..5}; do
            if [ "$j" -lt "$i" ]; then 
              prev_model="distillchain_v2_video_dcc_${j}"
              preensemble_data_pattern="${preensemble_data_pattern},/Youtube-8M/model_predictions_local/${part}/${model_name}/${prev_model}/*.tfrecord"
            fi
          done

          echo preensemble_data_pattern = "${preensemble_data_pattern}"
          CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble-with-predictions.py \
            --output_dir="$infer_data_path" \
            --train_dir="$sub_model_path" \
            --input_data_pattern="/Youtube-8M/data/video/${part}/*.tfrecord" \
            --distill_data_pattern="$preensemble_data_pattern" \
            --frame_features=False \
            --feature_names="mean_rgb,mean_audio" \
            --feature_sizes="1024,128" \
            --distillation_features=False \
            --distillation_as_input=True \
            --model=DistillchainDeepCombineChainModel \
            --moe_num_mixtures=4 \
            --deep_chain_layers=3 \
            --deep_chain_relu_cells=256 \
            --batch_size=1024 \
            --file_size=4096
        fi
      done
    fi

  done
fi

