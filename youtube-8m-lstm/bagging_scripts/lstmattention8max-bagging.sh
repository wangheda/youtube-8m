#!/bin/bash
DEFAULT_GPU_ID=1

if [ -z ${CUDA_VISIBLE_DEVICES+x} ]; then
  GPU_ID=$DEFAULT_GPU_ID
  echo "set CUDA_VISIBLE_DEVICES to default('$GPU_ID')"
else
  GPU_ID=$CUDA_VISIBLE_DEVICES
  echo "set CUDA_VISIBLE_DEVICES to external('$GPU_ID')"
fi

# base_model or sub_model_1 or sub_model_2 or so on
model_type="$1"

model_name="lstmattention8max_bagging"
MODEL_DIR="../model/${model_name}"

vocab_file="resources/train.video_id.vocab"
default_freq_file="resources/train.video_id.freq"

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

if [ $model_type == "base_model" ]; then

  # base model
  rm ${MODEL_DIR}/ensemble.conf
  mkdir -p $base_model_dir

  CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
    --train_dir="$base_model_dir" \
    --train_data_pattern="/Youtube-8M/data/frame/train/train*" \
    --frame_features=True \
    --feature_names="rgb,audio" \
    --feature_sizes="1024,128" \
    --reweight=True \
    --sample_vocab_file="$vocab_file" \
    --sample_freq_file="$default_freq_file" \
    --model=LstmAttentionMaxPoolingModel \
    --moe_num_mixtures=8 \
    --lstm_attentions=8 \
    --rnn_swap_memory=True \
    --base_learning_rate=0.001 \
    --num_readers=2 \
    --num_epochs=5 \
    --batch_size=128 \
    --keep_checkpoint_every_n_hour=0.5

elif [[ $model_type =~ ^sub_model ]]; then

  # sub model
  sub_model_dir="${MODEL_DIR}/${model_type}"

  if [ ! -d $sub_model_dir ]; then
    cp -r $base_model_dir $sub_model_dir

    # generate freq file
    python training_utils/sample_freq.py \
        --video_id_file="$vocab_file" \
        --output_freq_file="${sub_model_dir}/train.video_id.freq"

    # train N models with re-weighted samples
    CUDA_VISIBLE_DEVICES="$GPU_ID" python train-with-rebuild.py \
      --train_dir="$sub_model_dir" \
      --train_data_pattern="/Youtube-8M/data/frame/train/train*" \
      --frame_features=True \
      --feature_names="rgb,audio" \
      --feature_sizes="1024,128" \
      --reweight=True \
      --sample_vocab_file="resources/train.video_id.vocab" \
      --sample_freq_file="${sub_model_dir}/train.video_id.freq" \
      --model=LstmAttentionMaxPoolingModel \
      --moe_num_mixtures=8 \
      --lstm_attentions=8 \
      --rnn_swap_memory=True \
      --base_learning_rate=0.001 \
      --num_readers=4 \
      --num_epochs=1 \
      --batch_size=128 \
      --keep_checkpoint_every_n_hour=72.0 
  fi

  # inference-pre-ensemble
  for part in test ensemble_validate ensemble_train; do
    output_dir="/Youtube-8M/model_predictions/${part}/${model_name}/${model_type}"
    if [ ! -d $output_dir ]; then
      CUDA_VISIBLE_DEVICES="$GPU_ID" python inference-pre-ensemble.py \
        --output_dir="$output_dir" \
        --train_dir="${sub_model_dir}" \
        --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
        --frame_features=True \
        --feature_names="rgb,audio" \
        --feature_sizes="1024,128" \
        --model=LstmAttentionMaxPoolingModel \
        --moe_num_mixtures=8 \
        --lstm_attentions=8 \
        --rnn_swap_memory=True \
        --batch_size=32 \
        --file_size=4096
    fi
  done

  echo "${model_name}/${model_type}" >> ${MODEL_DIR}/ensemble.conf

fi

# on ensemble server
#cd ../youtube-8m-ensemble
#bash ensemble_scripts/eval-mean_model.sh ${model_name}/ensemble_mean_model ${MODEL_DIR}/ensemble.conf
#bash ensemble_scripts/infer-mean_model.sh ${model_name}/ensemble_mean_model ${MODEL_DIR}/ensemble.conf
