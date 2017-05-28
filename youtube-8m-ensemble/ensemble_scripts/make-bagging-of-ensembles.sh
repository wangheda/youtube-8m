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
model_type="$1"

model_name="boe_ensemble_matrix_model_no10"
MODEL_DIR="../model/${model_name}"
main_conf="ensemble_scripts/ensemble_no10.conf"

# generate vocab file
vocab_file="resources/validate.video_id.vocab"
default_freq_file="resources/validate.video_id.freq"

if [ ! -f $vocab_file ]; then
  cd resources
  wget http://us.data.yt8m.org/1/ground_truth_labels/validate_labels.csv
  echo "OOV" > validate.video_id.vocab
  cat validate_labels.csv | cut -d ',' -f 1 >> validate.video_id.vocab
  cd ..
fi

vocab_checksum=$(md5sum $vocab_file | cut -d ' ' -f 1)
if [ "$vocab_checksum" == "4a0c959434bb540f926d17a492161ca8" ]; then
  echo $vocab_file is valid
else
  echo $vocab_file is corrupted
  exit 1
fi

if [ ! -f $default_freq_file ]; then
  cat $vocab_file | awk '{print 1}' > $default_freq_file
fi

rm ${MODEL_DIR}/ensemble.conf
mkdir $MODEL_DIR

for i in {1..8}; do

  # sub model
  model_type="sub_model_${i}"
  sub_model_dir="${MODEL_DIR}/${model_type}"
  sub_conf="${sub_model_dir}/ensemble.conf"
  sub_freq="${sub_model_dir}/validate.video_id.freq"

  if [ ! -d $sub_model_dir ]; then
    mkdir -p $sub_model_dir

    # generate freq file
    python training_utils/sample_freq.py \
        --video_id_file="$vocab_file" \
        --output_freq_file="$sub_freq"

    # generate conf file
    python training_utils/sample_conf.py \
        --main_conf_file="$main_conf" \
        --sub_conf_file="$sub_conf"

    # train data patterns for ensemble training
    train_path=/Youtube-8M/model_predictions/ensemble_train
    sub_train_data_patterns=""
    for d in $(cat $sub_conf); do
      sub_train_data_patterns="${train_path}/${d}/*.tfrecord${sub_train_data_patterns:+,$sub_train_data_patterns}"
    done
    echo "$sub_train_data_patterns"

    # train N models with re-weighted samples
    CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
        --train_dir="$sub_model_dir" \
        --train_data_patterns="$sub_train_data_patterns" \
        --reweight=True \
        --sample_vocab_file="$vocab_file" \
        --sample_freq_file="$sub_freq" \
        --model=MatrixRegressionModel \
        --keep_checkpoint_every_n_hours=0.1 \
        --batch_size=1024 \
        --num_epochs=2

    # inference-pre-ensemble
    for part in test ensemble_validate ensemble_train; do
      output_dir="/Youtube-8M/model_predictions/${part}/${model_name}/${model_type}"
      if [ ! -d $output_dir ]; then
        # test data patterns
        test_path=/Youtube-8M/model_predictions/${part}
        sub_test_data_patterns=""
        for d in $(cat $sub_conf); do
          sub_test_data_patterns="${test_path}/${d}/*.tfrecord${sub_test_data_patterns:+,$sub_test_data_patterns}"
        done
        echo "$sub_test_data_patterns"

        CUDA_VISIBLE_DEVICES="$GPU_ID" python inference-pre-ensemble.py \
            --output_dir="$output_dir" \
            --train_dir="$sub_model_dir" \
            --input_data_patterns="$sub_test_data_patterns" \
            --model="MatrixRegressionModel" \
            --batch_size=1024 \
            --file_size=4096
      fi
    done
  fi

  echo "${model_name}/${model_type}" >> ${MODEL_DIR}/ensemble.conf
done

# on ensemble server
bash ensemble_scripts/train-attention_matrix_model.sh ${model_name}/ensemble_attention_matrix_model ${MODEL_DIR}/ensemble.conf
bash ensemble_scripts/eval-attention_matrix_model.sh ${model_name}/ensemble_attention_matrix_model ${MODEL_DIR}/ensemble.conf
# bash ensemble_scripts/infer-mean_model.sh ${model_name}/ensemble_mean_model ${MODEL_DIR}/ensemble.conf
