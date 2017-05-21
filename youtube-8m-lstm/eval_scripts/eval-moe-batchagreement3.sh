
GPU_ID=1
EVERY=100
MODEL=MoeModel
MODEL_DIR="../model/video_moe16_batchagreement0.01"

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
			--label_loss=BatchAgreementCrossEntropyLoss \
			--batch_agreement=0.01 \
			--batch_size=128 \
			--model=$MODEL \
			--moe_num_mixtures=16 \
			--run_once=True
	fi
done

