
GPU_ID="1"
EVERY=100
MODEL=ChainSupportReluMoeModel
MODEL_DIR="../model/video_chain_moe16_ce_freq"

start=$1
DIR="$(pwd)"

for checkpoint in $(cd $MODEL_DIR && python ${DIR}/training_utils/select.py $EVERY); do
	echo $checkpoint;
	if [ $checkpoint -gt $start ]; then
		echo $checkpoint;
		CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
			--train_dir="$MODEL_DIR" \
			--model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
			--eval_data_pattern="/Youtube-8M/data/video/validate/validatea*" \
			--frame_features=False \
			--feature_names="mean_rgb,mean_audio" \
			--feature_sizes="1024,128" \
			--batch_size=256 \
			--num_supports=200 \
			--support_type="frequent" \
			--model=$MODEL \
			--moe_num_mixtures=16 \
			--run_once=True
	fi
done

