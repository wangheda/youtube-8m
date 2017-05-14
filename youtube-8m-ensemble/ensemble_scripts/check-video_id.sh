model=$1
conf=$2
part=$3

validate_path=/Youtube-8M/model_predictions/${part}
validate_data_patterns=""
for d in $(cat $conf); do
  validate_data_patterns="${validate_path}/${d}/*.tfrecord${validate_data_patterns:+,$validate_data_patterns}"
done
echo "$validate_data_patterns"
input_data_pattern="${validate_path}/model_input/*.tfrecord"

CUDA_VISIBLE_DEVICES=""  python check_video_id.py \
    --input_data_pattern=$input_data_pattern \
    --eval_data_patterns="$validate_data_patterns"

