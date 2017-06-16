conf=$1
part=$2

validate_path=/Youtube-8M/model_predictions_local/${part}
validate_data_patterns=""
for d in $(cat $conf); do
  validate_data_patterns="${validate_path}/${d}/*.tfrecord${validate_data_patterns:+,$validate_data_patterns}"
done
echo "$validate_data_patterns"
input_data_pattern="/Youtube-8M/data/video/${part}/*.tfrecord"
#input_data_pattern="/Youtube-8M/model_predictions/${part}/model_input/*.tfrecord"

CUDA_VISIBLE_DEVICES=""  python check_video_id_match.py \
    --input_data_pattern=$input_data_pattern \
    --eval_data_patterns="$validate_data_patterns"

