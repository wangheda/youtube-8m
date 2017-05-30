#!/bin/bash

task=$1

model_name="hybridchain"
model_path="../model/${model_name}"

if [ $task == "train" ]; then
  predictions_data_pattern="/Youtube-8M/model_predictions/train/distillation/ensemble_v2_matrix_model/*.tfrecord"

  # train model 1: Video-DCC
  sub_model_name="distillchain_v2_video_dcc"
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

  # inference model 1
  infer_data_path="/Youtube-8M/model_predictions/train/${model_name}/${sub_model_name}"
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

  # next predictions pattern
  predictions_data_pattern="${predictions_data_pattern},${infer_data_path}/*.tfrecord"

  # train model 2: CNN-DCC
  sub_model_name="distillchain_v2_cnn_dcc"
  sub_model_path="${model_path}/${sub_model_name}"
  if [ ! -d $sub_model_path ]; then 
    mkdir -p $sub_model_path
    echo predictions = "$predictions_data_pattern"
    CUDA_VISIBLE_DEVICES=0 python train-with-predictions.py \
      --predictions_data_pattern="$predictions_data_pattern" \
      --distillation_features=True \
      --distillation_as_input=True \
      --train_dir="$sub_model_path" \
      --train_data_pattern="/Youtube-8M/data/frame/train/*.tfrecord" \
      --frame_features=True \
      --feature_names="rgb,audio" \
      --feature_sizes="1024,128" \
      --model=DistillchainCnnDeepCombineChainModel \
      --moe_num_mixtures=4 \
      --deep_chain_layers=3 \
      --deep_chain_relu_cells=256 \
      --multitask=True \
      --label_loss=MultiTaskCrossEntropyLoss \
      --support_type="label,label,label" \
      --support_loss_percent=0.05 \
      --num_readers=4 \
      --batch_size=128 \
      --num_epochs=2 \
      --keep_checkpoint_every_n_hours=20 \
      --base_learning_rate=0.001
  fi

  # inference model 2
  infer_data_path="/Youtube-8M/model_predictions/train/${model_name}/${sub_model_name}"
  if [ ! -d $infer_data_path ]; then
    CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble-with-predictions.py \
      --output_dir="$infer_data_path" \
      --train_dir="$sub_model_path" \
      --input_data_pattern="/Youtube-8M/data/frame/train/*.tfrecord" \
      --frame_features=True \
      --feature_names="rgb,audio" \
      --feature_sizes="1024,128" \
      --distill_data_pattern="$predictions_data_pattern" \
      --distillation_features=True \
      --distillation_as_input=True \
      --model=DistillchainCnnDeepCombineChainModel \
      --moe_num_mixtures=4 \
      --deep_chain_layers=3 \
      --deep_chain_relu_cells=256 \
      --batch_size=128 \
      --file_size=4096
  fi

  # next predictions pattern
  predictions_data_pattern="${predictions_data_pattern},${infer_data_path}/*.tfrecord"

  # train model 2: LSTM
  sub_model_name="distillchain_v2_lstmparalleloutput"
  sub_model_path="${model_path}/${sub_model_name}"
  if [ ! -d $sub_model_path ]; then 
    mkdir -p $sub_model_path
    echo predictions = "$predictions_data_pattern"
    CUDA_VISIBLE_DEVICES=0 python train-with-predictions.py \
      --predictions_data_pattern="$predictions_data_pattern" \
      --distillation_features=True \
      --distillation_as_input=True \
      --train_dir="$sub_model_path" \
      --train_data_pattern="/Youtube-8M/data/frame/train/*.tfrecord" \
      --frame_features=True \
      --feature_names="rgb,audio" \
      --feature_sizes="1024,128" \
      --model=DistillchainLstmParallelFinaloutputModel \
      --rnn_swap_memory=True \
	    --lstm_layers=1 \
	    --moe_num_mixtures=8 \
      --lstm_cells="1024,128" \
      --batch_size=128 \
      --num_epochs=1 \
      --keep_checkpoint_every_n_hours=20 \
      --base_learning_rate=0.0008
  fi

elif [ $task == "test-lstm" ]; then

  sub_model_name="distillchain_v2_lstmparalleloutput"
  sub_model_path="${model_path}/${sub_model_name}"
  if [ -d $sub_model_path ]; then 
    for part in ensemble_train ensemble_validate test; do
      infer_data_path="/Youtube-8M/model_predictions/${part}/${model_name}/${sub_model_name}"
      if [ ! -d $infer_data_path ]; then
        predictions_data_pattern="/Youtube-8M/model_predictions/${part}/distillation/ensemble_v2_matrix_model/*.tfrecord"
        for prev_model in distillchain_v2_video_dcc distillchain_v2_cnn_dcc; do 
          predictions_data_pattern="${predictions_data_pattern},/Youtube-8M/model_predictions/${part}/${model_name}/${prev_model}/*.tfrecord"
        done

        echo predictions = "$predictions_data_pattern"
        CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble-with-predictions.py \
          --output_dir="$infer_data_path" \
          --train_dir="$sub_model_path" \
          --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
          --frame_features=True \
          --feature_names="rgb,audio" \
          --feature_sizes="1024,128" \
          --distill_data_pattern="$predictions_data_pattern" \
          --distillation_features=True \
          --distillation_as_input=True \
	        --model=DistillchainLstmParallelFinaloutputModel \
          --rnn_swap_memory=True \
          --lstm_layers=1 \
          --moe_num_mixtures=8 \
          --lstm_cells="1024,128" \
          --batch_size=128 \
          --file_size=4096
      fi
    done
  fi

elif [ $task == "test-cnn" ]; then

  sub_model_name="distillchain_v2_cnn_dcc"
  sub_model_path="${model_path}/${sub_model_name}"
  if [ -d $sub_model_path ]; then 
    for part in ensemble_train ensemble_validate test; do
      infer_data_path="/Youtube-8M/model_predictions/${part}/${model_name}/${sub_model_name}"
      if [ ! -d $infer_data_path ]; then
        predictions_data_pattern="/Youtube-8M/model_predictions/${part}/distillation/ensemble_v2_matrix_model/*.tfrecord"
        for prev_model in distillchain_v2_video_dcc; do 
          predictions_data_pattern="${predictions_data_pattern},/Youtube-8M/model_predictions/${part}/${model_name}/${prev_model}/*.tfrecord"
        done

        echo predictions = "$predictions_data_pattern"
        CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble-with-predictions.py \
          --output_dir="$infer_data_path" \
          --train_dir="$sub_model_path" \
          --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
          --frame_features=True \
          --feature_names="rgb,audio" \
          --feature_sizes="1024,128" \
          --distill_data_pattern="$predictions_data_pattern" \
          --distillation_features=True \
          --distillation_as_input=True \
          --model=DistillchainCnnDeepCombineChainModel \
          --moe_num_mixtures=4 \
          --deep_chain_layers=3 \
          --deep_chain_relu_cells=256 \
          --batch_size=128 \
          --file_size=4096
      fi
    done
  fi

elif [ $task == "test-video" ]; then

  sub_model_name="distillchain_v2_video_dcc"
  sub_model_path="${model_path}/${sub_model_name}"
  if [ -d $sub_model_path ]; then 
    for part in ensemble_train ensemble_validate test; do
      infer_data_path="/Youtube-8M/model_predictions/${part}/${model_name}/${sub_model_name}"
      if [ ! -d $infer_data_path ]; then
        predictions_data_pattern="/Youtube-8M/model_predictions/${part}/distillation/ensemble_v2_matrix_model/*.tfrecord"

        echo predictions = "$predictions_data_pattern"
        CUDA_VISIBLE_DEVICES=1 python inference-pre-ensemble-with-predictions.py \
          --output_dir="$infer_data_path" \
          --train_dir="$sub_model_path" \
          --input_data_pattern="/Youtube-8M/data/video/${part}/*.tfrecord" \
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
    done
  fi

fi

