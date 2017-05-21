
GPU_ID=1
EVERY=500
MODEL=FramehopLstmMemoryModel
MODEL_DIR="../model/framehop_lstm"

start=$1
DIR="$(pwd)"

for checkpoint in $(cd $MODEL_DIR && python ${DIR}/training_utils/select.py $EVERY); do
	echo $checkpoint;
	if [ $checkpoint -gt $start ]; then
		echo $checkpoint;
		CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
			--train_dir="$MODEL_DIR" \
			--model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
			--eval_data_pattern="/Youtube-8M-validate/validatea*" \
			--frame_features=True \
			--feature_names="rgb,audio" \
			--feature_sizes="1024,128" \
			--batch_size=64 \
			--model=$MODEL \
			--deep_chain_layers=4 \
			--deep_chain_relu_cells=256 \
			--feature_transformer=IdenticalTransformer \
			--lstm_layers=2 \
			--lstm_cells="512,64" \
			--moe_num_mixtures=4 \
			--rnn_swap_memory=True \
			--num_readers=1 \
			--run_once=True
	fi
done

