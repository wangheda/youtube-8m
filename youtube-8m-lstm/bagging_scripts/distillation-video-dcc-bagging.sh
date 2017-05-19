#!/bin/bash

DEFAULT_GPU_ID=0

if [ -z ${CUDA_VISIBLE_DEVICES+x} ]; then 
  GPU_ID=$DEFAULT_GPU_ID
  echo "set CUDA_VISIBLE_DEVICES to default('$GPU_ID')"
else 
  GPU_ID=$CUDA_VISIBLE_DEVICES
  echo "set CUDA_VISIBLE_DEVICES to external('$GPU_ID')"
fi

model_name="distillation_video_dcc_bagging"
MODEL_DIR="../model/${model_name}"
rm ${MODEL_DIR}/ensemble.conf

vocab_file="resources/train.video_id.vocab"

if [ ! -f $vocab_file ]; then
  cd resources
  wget http://us.data.yt8m.org/1/ground_truth_labels/train_labels.csv
  echo "OOV" > train.video_id.vocab
  cat train_labels.csv | cut -d ',' -f 1 >> train.video_id.vocab
  cd ..
fi

vocab_checksum=$(md5sum $vocab_file | cut -d ' ' -f 1)
if [ "$vocab_checksum" == "b74b8f2592cad5dd21bf614d1438db98" ]; then
  echo $vocab_file is valid
else
  echo $vocab_file is corrupted
  exit 1
fi

if [ ! -f $default_freq_file ]; then
  cat $vocab_file | awk '{print 1}' > $default_freq_file
fi

for i in {1..4}; do
  sub_model_dir="${MODEL_DIR}/sub_model_${i}"
  if [ ! -d $sub_model_dir ]; then
    mkdir -p $sub_model_dir

    # generate freq file
    if [ ! -f ${sub_model_dir}/train.video_id.freq ]; then
      python training_utils/sample_freq.py \
          --video_id_file="$vocab_file" \
          --output_freq_file="${sub_model_dir}/train.video_id.freq"
    fi

    # train N models with re-weighted samples
    CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
      --train_dir="$sub_model_dir" \
      --train_data_pattern="/Youtube-8M/distillation/video/train/*.tfrecord" \
      --frame_features=False \
      --feature_names="mean_rgb,mean_audio" \
      --feature_sizes="1024,128" \
      --distillation_features=True \
      --distillation_type=1 \
      --distillation_percent=0.5 \
      --model=DeepCombineChainModel \
      --moe_num_mixtures=4 \
      --deep_chain_relu_cells=256 \
      --deep_chain_layers=4 \
      --multitask=True \
      --label_loss=MultiTaskCrossEntropyLoss \
      --support_type="label,label,label,label" \
      --num_supports=18864 \
      --support_loss_percent=0.05 \
      --reweight=True \
      --sample_vocab_file="resources/train.video_id.vocab" \
      --sample_freq_file="${sub_model_dir}/train.video_id.freq" \
      --keep_checkpoint_every_n_hour=10 \
      --keep_checkpoint_interval=6 \
      --base_learning_rate=0.01 \
      --data_augmenter=NoiseAugmenter \
      --input_noise_level=0.1 \
      --num_readers=4 \
      --num_epochs=6 \
      --batch_size=512
  fi

  # inference-pre-ensemble
  for part in test ensemble_validate ensemble_train; do
    output_dir="/Youtube-8M/model_predictions/${part}/${model_name}/sub_model_$i"
    if [ ! -d $output_dir ]; then
      CUDA_VISIBLE_DEVICES="$GPU_ID" python inference-pre-ensemble.py \
        --output_dir="$output_dir" \
        --train_dir="${sub_model_dir}" \
        --input_data_pattern="/Youtube-8M/data/video/${part}/*.tfrecord" \
        --frame_features=False \
        --feature_names="mean_rgb,mean_audio" \
        --feature_sizes="1024,128" \
        --model=DeepCombineChainModel \
        --moe_num_mixtures=4 \
        --deep_chain_relu_cells=256 \
        --deep_chain_layers=4 \
        --batch_size=1024 \
        --file_size=4096
    fi
  done

  echo "${model_name}/sub_model_$i" >> ${MODEL_DIR}/ensemble.conf

done

#cd ../youtube-8m-ensemble
#bash ensemble_scripts/eval-mean_model.sh video_bagging/ensemble_mean_model ${MODEL_DIR}/ensemble.conf
#bash ensemble_scripts/infer-mean_model.sh video_bagging/ensemble_mean_model ${MODEL_DIR}/ensemble.conf
