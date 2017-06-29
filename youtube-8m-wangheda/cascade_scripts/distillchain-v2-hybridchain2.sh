#!/bin/bash

task=$1

model_name="hybridchain2"
model_path="../model/${model_name}"

if [ $task == "train" ]; then
  predictions_data_pattern="/Youtube-8M/model_predictions_x32/train/distillation/ensemble_v2_matrix_model/*.tfrecord"

  # train model 1: LSTM
  sub_model_name="distillchain_v2_lstmparalleloutput"
  sub_model_path="${model_path}/${sub_model_name}"
  if [ ! -d $sub_model_path ]; then 
    mkdir -p $sub_model_path
    echo predictions = "$predictions_data_pattern"
    CUDA_VISIBLE_DEVICES=0 python train-with-predictions.py \
      --predictions_data_pattern="$predictions_data_pattern" \
      --distillation_features=False \
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

  # inference model 2
  infer_data_path="/Youtube-8M/model_predictions_local/train/${model_name}/${sub_model_name}"
  if [ ! -d $infer_data_path ]; then
    CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble-with-predictions.py \
      --output_dir="$infer_data_path" \
      --train_dir="$sub_model_path" \
      --input_data_pattern="/Youtube-8M/data/frame/train/*.tfrecord" \
      --frame_features=True \
      --feature_names="rgb,audio" \
      --feature_sizes="1024,128" \
      --distill_data_pattern="$predictions_data_pattern" \
      --distillation_features=False \
      --distillation_as_input=True \
      --model=DistillchainLstmParallelFinaloutputModel \
      --rnn_swap_memory=True \
	    --lstm_layers=1 \
	    --moe_num_mixtures=8 \
      --lstm_cells="1024,128" \
      --batch_size=128 \
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
      --distillation_features=False \
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
      --num_epochs=1 \
      --keep_checkpoint_every_n_hours=20 \
      --base_learning_rate=0.001
  fi


elif [ $task == "test-lstm" ]; then

  sub_model_name="distillchain_v2_lstmparalleloutput"
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
          --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
          --frame_features=True \
          --feature_names="rgb,audio" \
          --feature_sizes="1024,128" \
          --distill_data_pattern="$predictions_data_pattern" \
          --distillation_features=False \
          --distillation_as_input=True \
	        --model=DistillchainLstmParallelFinaloutputModel \
          --rnn_swap_memory=True \
          --lstm_layers=1 \
          --moe_num_mixtures=8 \
          --lstm_cells="1024,128" \
          --batch_size=32 \
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
        for prev_model in distillchain_v2_lstmparalleloutput; do 
          predictions_data_pattern="${predictions_data_pattern},/Youtube-8M/model_predictions_local/${part}/${model_name}/${prev_model}/*.tfrecord"
        done

        echo predictions = "$predictions_data_pattern"
        CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble-with-predictions.py \
          --output_dir="$infer_data_path" \
          --train_dir="$sub_model_path" \
          --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
          --frame_features=True \
          --feature_names="rgb,audio" \
          --feature_sizes="1024,128" \
          --distill_data_pattern="$predictions_data_pattern" \
          --distillation_features=False \
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

fi

