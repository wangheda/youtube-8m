
GPU_ID=1
EVERY=500
MODEL=LstmMemoryDeepChainModel
MODEL_DIR="../model/multilstmmemory1024_moe4_deep_chain"

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
			--batch_size=64 \
			--model=$MODEL \
			--moe_num_mixtures=4 \
			--deep_chain_relu_cells=200 \
			--deep_chain_layers=1 \
			--lstm_layers=2 \
			--lstm_cells=1024 \
			--multitask=True \
			--support_type="label" \
			--num_supports=4716 \
			--label_loss=MultiTaskCrossEntropyLoss \
			--support_loss_percent=0.2 \
			--rnn_swap_memory=True \
			--run_once=True
	fi
done

