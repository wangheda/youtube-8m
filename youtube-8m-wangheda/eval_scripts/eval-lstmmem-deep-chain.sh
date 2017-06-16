
GPU_ID=1
EVERY=500
MODEL=LstmMemoryModel
MODEL_DIR="../model/lstmmemory1024_moe4_deep_chain"

start=$1
DIR="$(pwd)"

for checkpoint in $(cd $MODEL_DIR && python ${DIR}/training_utils/select.py $EVERY); do
	echo $checkpoint;
	if [ $checkpoint -gt $start ]; then
		echo $checkpoint;
		CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
			--train_dir="$MODEL_DIR" \
			--model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
			--eval_data_pattern="/run/shm/validate/validatea*" \
			--frame_features=True \
			--feature_names="rgb,audio" \
			--feature_sizes="1024,128" \
			--model=LstmMemoryModel \
			--lstm_layers=2 \
			--lstm_cells=1024 \
			--video_level_classifier_model=DeepChainModel \
			--deep_chain_relu_cells=200 \
			--deep_chain_layers=1 \
			--moe_num_mixtures=4 \
			--label_loss=MultiTaskCrossEntropyLoss \
			--multitask=True \
			--support_type="label" \
			--num_supports=4716 \
			--support_loss_percent=0.1 \
			--batch_size=128 \
			--num_readers=4 \
			--run_once=True
	fi
done

