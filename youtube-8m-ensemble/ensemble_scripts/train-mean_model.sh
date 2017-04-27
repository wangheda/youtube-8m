model=$1
conf=$2
train_path=/Youtube-8M/model_predictions/ensemble_train

train_data_patterns=""
for d in $(cat $conf); do
  train_data_patterns="${train_path}/${d}/*.tfrecord${train_data_patterns:+,$train_data_patterns}"
done
echo $train_data_patterns

CUDA_VISIBLE_DEVICES=0 python train.py \
      --train_dir="../model/${model}" \
      --train_data_patterns="$train_data_patterns" \
      --model=MeanModel \
      --training=False \
      --num_epochs=1
