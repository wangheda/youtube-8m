#!/bin/bash

model_name="lstmparalleloutput_bagging"
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

for i in {1..2}; do
  sub_model_dir="${MODEL_DIR}/sub_model_${i}"
  mkdir -p $sub_model_dir

  # generate freq file
  python training_utils/sample_freq.py \
      --video_id_file="$vocab_file" \
      --output_freq_file="${sub_model_dir}/train.video_id.freq"

  # train N models with re-weighted samples
  CUDA_VISIBLE_DEVICES=1 python train.py \
    --train_dir="$sub_model_dir" \
    --train_data_pattern="/Youtube-8M/data/frame/train/train*" \
    --frame_features=True \
    --feature_names="rgb,audio" \
    --feature_sizes="1024,128" \
    --reweight=True \
    --sample_vocab_file="resources/train.video_id.vocab" \
    --sample_freq_file="${sub_model_dir}/train.video_id.freq" \
    --model=LstmParallelFinaloutputModel \
    --lstm_cells="1024,128" \
    --moe_num_mixtures=8 \
    --rnn_swap_memory=True \
    --base_learning_rate=0.001 \
    --num_readers=2 \
    --num_epochs=3 \
    --batch_size=96 \
    --keep_checkpoint_every_n_hour=72.0 

  # inference-pre-ensemble
  for part in test ensemble_validate ensemble_train; do
    CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble.py \
      --output_dir="/Youtube-8M/model_predictions/${part}/${model_name}/sub_model_$i" \
      --train_dir="${sub_model_dir}" \
      --input_data_pattern="/Youtube-8M/data/frame/${part}/*.tfrecord" \
      --frame_features=True \
      --feature_names="rgb,audio" \
      --feature_sizes="1024,128" \
      --batch_size=32 \
      --file_size=4096
  done

  echo "${model_name}/sub_model_$i" >> ${MODEL_DIR}/ensemble.conf

done

# on ensemble server
#cd ../youtube-8m-ensemble
#bash ensemble_scripts/eval-mean_model.sh ${model_name}/ensemble_mean_model ${MODEL_DIR}/ensemble.conf
#bash ensemble_scripts/infer-mean_model.sh ${model_name}/ensemble_mean_model ${MODEL_DIR}/ensemble.conf
