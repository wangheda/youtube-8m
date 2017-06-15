#!/bin/bash

# the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${DIR}/../

model_name="distillation/video_chain_moe2_verydeep_combine"
checkpoint="28403"
for part in ensemble_validate test ensemble_train; do 
  output_dir="/Youtube-8M/model_predictions/${part}/${model_name}"
  if [ ! -d $output_dir ]; then 
    CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble.py \
      --output_dir="$output_dir" \
      --model_checkpoint_path="../model/${model_name}/model.ckpt-${checkpoint}" \
      --input_data_pattern="/Youtube-8M/data/video/${part}/*.tfrecord" \
      --frame_features=False \
      --feature_names="mean_rgb,mean_audio" \
      --feature_sizes="1024,128" \
      --model=DeepCombineChainModel \
      --deep_chain_relu_cells=128 \
      --deep_chain_layers=8 \
      --moe_num_mixtures=2 \
      --batch_size=1024 \
      --file_size=4096
  fi
done

model_name="distillation/cnn_deep_combine_chain"
checkpoint="228332"
for part in ensemble_validate test ensemble_train; do 
  output_dir="/Youtube-8M/model_predictions/${part}/${model_name}"
  if [ ! -d $output_dir ]; then 
    CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble.py \
      --output_dir="$output_dir" \
      --model_checkpoint_path="../model/${model_name}/model.ckpt-${checkpoint}" \
      --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
      --frame_features=True \
      --feature_names="rgb,audio" \
      --feature_sizes="1024,128" \
      --model=CnnDeepCombineChainModel \
      --deep_chain_layers=4 \
      --deep_chain_relu_cells=128 \
      --moe_num_mixtures=4 \
      --batch_size=128 \
      --file_size=4096
  fi
done

model_name="distillation/lstmparallelfinaloutput1024_moe8"
checkpoint="144351"
for part in ensemble_validate test ensemble_train; do 
  output_dir="/Youtube-8M/model_predictions/${part}/${model_name}"
  if [ ! -d $output_dir ]; then 
    CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble.py \
      --output_dir="$output_dir" \
      --model_checkpoint_path="../model/${model_name}/model.ckpt-${checkpoint}" \
      --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
      --frame_features=True \
      --feature_names="rgb,audio" \
      --feature_sizes="1024,128" \
      --model=LstmParallelFinaloutputModel \
      --lstm_cells="1024,128" \
      --moe_num_mixtures=8 \
      --rnn_swap_memory=True \
      --batch_size=128 \
      --file_size=4096
  fi
done

# The Lstm-Attention-Max-pooling model
cd ${DIR}/../../youtube-8m-zhangteng/
model_name="distillation/frame_level_lstm_extend8_model"
checkpoint="181785"
for part in ensemble_validate test ensemble_train; do 
  output_dir="/Youtube-8M/model_predictions/${part}/${model_name}"
  if [ ! -d $output_dir ]; then 
    CUDA_VISIBLE_DEVICES=0 python inference_with_rebuild.py \
      --output_dir="$output_dir" \
      --model_checkpoint_path="../model/${model_name}/model.ckpt-${checkpoint}" \
      --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
      --frame_features=True \
      --feature_names="rgb,audio" \
      --feature_sizes="1024,128" \
      --model=LstmExtendModel \
      --video_level_classifier_model=MoeExtendModel \
      --moe_num_mixtures=8 \
      --moe_num_extend=8 \
      --batch_size=128 \
      --file_size=4096
  fi
done
