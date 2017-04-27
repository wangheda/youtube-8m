model=$1
conf=$2
test_path=/Youtube-8M/model_predictions/test

test_data_patterns=""
for d in $(cat $conf); do
  test_data_patterns="${test_path}/${d}/*.tfrecord${test_data_patterns:+,$test_data_patterns}"
done
echo "$test_data_patterns"

#CUDA_VISIBLE_DEVICES=0 python inference.py \
python inference.py \
      --model_checkpoint_path="../model/${model}/model.ckpt-1094" \
      --output_file="../model/${model}/predictions.${model}.csv" \
      --model="MatrixRegressionModel" \
      --input_data_patterns="$test_data_patterns"
