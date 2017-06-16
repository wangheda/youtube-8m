GPU_ID=0
EVERY=500
MODEL=LstmCnnDeepCombineChainModel
MODEL_DIR="../model/lstm_cnn_deep_combine_chain"

start=$1
DIR="$(pwd)"

for checkpoint in $(cd $MODEL_DIR && python ${DIR}/training_utils/select.py $EVERY); do
  echo $checkpoint;
  if [ $checkpoint -gt $start ]; then
    echo $checkpoint;
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
      --train_dir="$MODEL_DIR" \
      --model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
      --eval_data_pattern="/Youtube-8M/data/frame/validate/validatea*" \
      --frame_features=True \
      --feature_names="rgb,audio" \
      --feature_sizes="1024,128" \
      --model=$MODEL \
      --lstm_layers=1 \
      --lstm_cells="1024,128" \
      --deep_chain_layers=3 \
      --deep_chain_relu_cells=128 \
      --moe_num_mixtures=4 \
      --label_loss=MultiTaskCrossEntropyLoss \
      --multitask=True \
      --support_type="label,label,label" \
      --support_loss_percent=0.05 \
      --batch_size=32 \
      --rnn_swap_memory=True \
      --num_readers=4 \
      --run_once=True
  fi
done


