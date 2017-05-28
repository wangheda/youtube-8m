model=$1
conf=$2
checkpoint=$3
test_path=/Youtube-8M/model_predictions/test

test_data_patterns=""
for d in $(cat $conf); do
  test_data_patterns="${test_path}/${d}/*.tfrecord${test_data_patterns:+,$test_data_patterns}"
done
echo "$test_data_patterns"
input_data_pattern="${test_path}/model_input/*.tfrecord"

#CUDA_VISIBLE_DEVICES=0 python inference.py \
python inference.py \
      --model_checkpoint_path="../model/${model}/model.ckpt-${checkpoint}" \
      --output_file="../model/${model}/predictions.${model}.csv" \
      --model="AttentionMatrixModel" \
      --moe_num_mixtures=4 \
      --attention_relu_cells=256 \
      --attention_matrix_rank=4 \
      --batch_size=1024 \
      --input_data_pattern="$input_data_pattern" \
      --input_data_patterns="$test_data_patterns"
