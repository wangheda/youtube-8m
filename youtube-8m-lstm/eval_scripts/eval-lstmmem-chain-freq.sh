GPU_ID=1
EVERY=500
MODEL=LstmMemoryModel
MODEL_DIR="../model/lstmmemory1024_moe8_chain_freq"

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
			--model=LstmMemoryModel \
			--video_level_classifier_model=ChainSupportReluMoeModel \
			--lstm_layers=2 \
			--lstm_cells=1024 \
			--multitask=True \
			--support_type="frequent" \
			--num_supports=200 \
			--num_frequents=200 \
			--label_loss=MultiTaskCrossEntropyLoss \
			--base_learning_rate=0.0008 \
			--vertical_loss_percent=0.4 \
			--moe_num_mixtures=8 \
			--num_readers=1 \
			--run_once=True
	fi
done

