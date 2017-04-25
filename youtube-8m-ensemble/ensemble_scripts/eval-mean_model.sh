train_path=/Youtube-8M/model_predictions/ensemble_train
train_data_patterns=""
for d in $(ls $train_path | sort); do
  train_data_patterns="${train_path}/${d}/*.tfrecord${train_data_patterns:+,$train_data_patterns}"
done
echo $train_data_patterns

CUDA_VISIBLE_DEVICES=0 python eval.py \
      --model_checkpoint_path="../model/mean_model/model.ckpt-0" \
      --train_dir="../model/mean_model" \
      --model="MeanModel" \
      --eval_data_patterns=$train_data_patterns
