#!/bin/bash
DEFAULT_GPU_ID=0

if [ -z ${CUDA_VISIBLE_DEVICES+x} ]; then
  GPU_ID=$DEFAULT_GPU_ID
  echo "set CUDA_VISIBLE_DEVICES to default('$GPU_ID')"
else
  GPU_ID=$CUDA_VISIBLE_DEVICES
  echo "set CUDA_VISIBLE_DEVICES to external('$GPU_ID')"
fi

# base_model or sub_model_1 or sub_model_2 or so on
#model_type="$1"

model_name="video_dcc_boosting"
MODEL_DIR="../model/${model_name}"

vocab_file="resources/train.video_id.vocab"
default_freq_file="resources/train.video_id.freq"

rm ${MODEL_DIR}/ensemble.conf
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

base_model_dir="${MODEL_DIR}/base_model"

# base model (4 epochs)
mkdir -p $base_model_dir
for j in {1..2}; do 
  CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
    --train_dir="$base_model_dir" \
    --train_data_pattern="/Youtube-8M/data/video/train/train*" \
    --frame_features=False \
    --feature_names="mean_rgb,mean_audio" \
    --feature_sizes="1024,128" \
    --model=DeepCombineChainModel \
    --moe_num_mixtures=4 \
    --deep_chain_relu_cells=256 \
    --deep_chain_layers=4 \
    --label_loss=MultiTaskCrossEntropyLoss \
    --multitask=True \
    --support_type="label,label,label,label" \
    --num_supports=18864 \
    --support_loss_percent=0.05 \
    --reweight=True \
    --sample_vocab_file="$vocab_file" \
    --sample_freq_file="$default_freq_file" \
    --keep_checkpoint_every_n_hour=8.0 \
    --keep_checkpoint_interval=6 \
    --base_learning_rate=0.01 \
    --data_augmenter=NoiseAugmenter \
    --input_noise_level=0.2 \
    --num_readers=2 \
    --num_epochs=2 \
    --batch_size=1024
done

last_freq_file=$default_freq_file

for i in {1..8}; do
  sub_model_dir="${MODEL_DIR}/sub_model_${i}"
  cp -r $base_model_dir $sub_model_dir

  echo "training model #$i, reweighting with $last_freq_file"
  # train N models with re-weighted samples
  CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
    --train_dir="$sub_model_dir" \
    --train_data_pattern="/Youtube-8M/data/video/train/train*" \
    --frame_features=False \
    --feature_names="mean_rgb,mean_audio" \
    --feature_sizes="1024,128" \
    --model=DeepCombineChainModel \
    --moe_num_mixtures=4 \
    --deep_chain_relu_cells=256 \
    --deep_chain_layers=4 \
    --label_loss=MultiTaskCrossEntropyLoss \
    --multitask=True \
    --support_type="label,label,label,label" \
    --num_supports=18864 \
    --support_loss_percent=0.05 \
    --reweight=True \
    --sample_vocab_file="$vocab_file" \
    --sample_freq_file="$last_freq_file" \
    --keep_checkpoint_every_n_hour=8.0 \
    --base_learning_rate=0.01 \
    --data_augmenter=NoiseAugmenter \
    --input_noise_level=0.2 \
    --num_readers=2 \
    --num_epochs=2 \
    --batch_size=1024

  # inference-pre-ensemble
  for part in test ensemble_validate ensemble_train; do
    CUDA_VISIBLE_DEVICES="$GPU_ID" python inference-pre-ensemble.py \
      --output_dir="/Youtube-8M/model_predictions/${part}/${model_name}/sub_model_$i" \
      --train_dir="${sub_model_dir}" \
      --input_data_pattern="/Youtube-8M/data/video/${part}/*.tfrecord" \
      --frame_features=False \
      --feature_names="mean_rgb,mean_audio" \
      --feature_sizes="1024,128" \
      --batch_size=128 \
      --file_size=4096
  done

  # get error mapping
  CUDA_VISIBLE_DEVICES="$GPU_ID" python inference-sample-error.py \
    --output_file="${sub_model_dir}/train.video_id.error" \
    --train_dir="${sub_model_dir}" \
    --input_data_pattern="/Youtube-8M/data/video/train/*.tfrecord" \
    --frame_features=False \
    --feature_names="mean_rgb,mean_audio" \
    --feature_sizes="1024,128" \
    --model=DeepCombineChainModel \
    --moe_num_mixtures=4 \
    --deep_chain_relu_cells=256 \
    --deep_chain_layers=4 \
    --batch_size=1024 

  # generate resample freq file
  python training_utils/reweight_sample_freq.py \
      --video_id_file="$vocab_file" \
      --input_freq_file="$last_freq_file" \
      --input_error_file="${sub_model_dir}/train.video_id.error" \
      --output_freq_file="${sub_model_dir}/train.video_id.next_freq"

  last_freq_file="${sub_model_dir}/train.video_id.next_freq"

  echo "${model_name}/sub_model_$i" >> ${MODEL_DIR}/ensemble.conf
done

cd ../youtube-8m-ensemble
CUDA_VISIBLE_DEVICES="$GPU_ID" bash ensemble_scripts/train-matrix_model.sh ${model_name}/ensemble_matrix_model ${MODEL_DIR}/ensemble.conf
CUDA_VISIBLE_DEVICES="$GPU_ID" bash ensemble_scripts/eval-matrix_model.sh ${model_name}/ensemble_matrix_model ${MODEL_DIR}/ensemble.conf
#CUDA_VISIBLE_DEVICES="$GPU_ID" bash ensemble_scripts/infer-matrix_model.sh ${model_name}/ensemble_matrix_model ${MODEL_DIR}/ensemble.conf

