model=$1
conf=$2
part=$3

test_path=/Youtube-8M/model_predictions/${part}

test_data_patterns=""
for d in $(cat $conf); do
  test_data_patterns="${test_path}/${d}/*.tfrecord${test_data_patterns:+,$test_data_patterns}"
done
echo "$test_data_patterns"

python inference-pre-ensemble.py \
	      --output_dir="/Youtube-8M/model_predictions/${part}/${model}" \
        --model_checkpoint_path="../model/${model}/model.ckpt-0" \
        --input_data_patterns="$test_data_patterns" \
        --model="MeanModel" \
	      --batch_size=1024 \
	      --file_size=4096
