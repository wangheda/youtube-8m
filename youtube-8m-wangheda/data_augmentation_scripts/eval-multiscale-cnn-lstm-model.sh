
GPU_ID="0"
EVERY=1000
MODEL=MultiscaleCnnLstmModel
MODEL_DIR="../model/dataaugmentation_multiscale_cnn_lstm"

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
    			--multiscale_cnn_lstm_layers=4 \
    			--moe_num_mixtures=4 \
			--run_once=True
	fi
done

