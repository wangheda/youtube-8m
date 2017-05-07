model=$1
conf=$2
part=$3
checkpoint=2316
test_path=/Youtube-8M/model_predictions/${part}

test_data_patterns=""
for d in $(cat $conf); do
  test_data_patterns="${test_path}/${d}/*.tfrecord${test_data_patterns:+,$test_data_patterns}"
done
echo "$test_data_patterns"

CUDA_VISIBLE_DEVICES=0 python inference-pre-ensemble.py \
	    --output_dir="/Youtube-8M/model_predictions/${part}/${model}" \
      --model_checkpoint_path="../model/${model}/model.ckpt-${checkpoint}" \
      --input_data_patterns="$test_data_patterns" \
      --model="MatrixRegressionModel" \
      --batch_size=1024 \
      --file_size=4096
