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

validate_path=/Youtube-8M/model_predictions/ensemble_validate
validate_data_patterns=""
for d in $(cat $conf); do
  validate_data_patterns="${validate_path}/${d}/*.tfrecord${validate_data_patterns:+,$validate_data_patterns}"
done
echo "$validate_data_patterns"

CUDA_VISIBLE_DEVICES="$GPU_ID" python eval.py \
      --model_checkpoint_path="../model/${model}/model.ckpt-0" \
      --train_dir="../model/${model}" \
      --model="MeanModel" \
      --eval_data_patterns="$validate_data_patterns"
