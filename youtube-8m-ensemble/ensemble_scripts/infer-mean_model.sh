test_path=/Youtube-8M/model_predictions/test
models=( "lstmmem1024_layer2_moe4_deep_combine_chain_add_length" "lstmmemory_cell1024_layer2_moe8" "video_very_deep_combine_chain" )

test_data_patterns=""
for d in ${models[@]}; do
  test_data_patterns="${test_path}/${d}/*.tfrecord${test_data_patterns:+,$test_data_patterns}"
done
echo "$test_data_patterns"
exit 0

CUDA_VISIBLE_DEVICES=0 python infer.py \
      --model_checkpoint_path="../model/mean_model/model.ckpt-0" \
      --train_dir="../model/mean_model" \
      --output_file="../model/mean_model/predictions.no1.csv" \
      --model="MeanModel" \
      --input_data_patterns=$test_data_patterns
