model=$1
conf=$2

DEFAULT_GPU_ID=0
if [ -z ${CUDA_VISIBLE_DEVICES+x} ]; then
  GPU_ID=$DEFAULT_GPU_ID
  echo "set CUDA_VISIBLE_DEVICES to default('$GPU_ID')"
else
  GPU_ID=$CUDA_VISIBLE_DEVICES
  echo "set CUDA_VISIBLE_DEVICES to external('$GPU_ID')"
fi

train_path=/Youtube-8M/model_predictions/ensemble_train
train_data_patterns=""
for d in $(cat $conf); do
  train_data_patterns="${train_path}/${d}/*.tfrecord${train_data_patterns:+,$train_data_patterns}"
done
echo $train_data_patterns
input_data_pattern="${train_path}/model_input/*.tfrecord"

CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
      --train_dir="../model/${model}" \
      --train_data_patterns="$train_data_patterns" \
      --input_data_pattern="$input_data_pattern" \
      --model=AttentionMatrixModel \
      --base_learning_rate=0.01 \
      --moe_num_mixtures=4 \
      --attention_relu_cells=256 \
      --attention_matrix_rank=4 \
      --keep_checkpoint_every_n_hours=0.1 \
      --batch_size=1024 \
      --num_epochs=4
