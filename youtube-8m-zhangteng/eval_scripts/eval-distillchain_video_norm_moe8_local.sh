GPU_ID=0
EVERY=1000
MODEL=MoeDistillChainNorm2Model
MODEL_DIR="../model/video_level_distillchainnorm2_model"
start=0
DIR="$(pwd)"

for checkpoint in $(cd $MODEL_DIR && python ${DIR}/training_utils/select.py $EVERY); do
 	echo $checkpoint;
	if [[ $checkpoint -gt $start ]]; then

 		echo $checkpoint;
		CUDA_VISIBLE_DEVICES=$GPU_ID python eval_distill.py \
 			--train_dir="$MODEL_DIR" \
			--model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
			--eval_data_pattern="/Youtube-8M/data/video/validate/validatea*" \
			--distill_data_pattern="/Youtube-8M/model_predictions/validatea/distillation/ensemble_mean_model/*.tfrecord" \
			--frame_features=False \
			--feature_names="mean_rgb,mean_audio" \
			--distill_names="predictions" \
			--feature_sizes="1024,128" \
			--distill_sizes="4716" \
			--batch_size=64 \
			--model=$MODEL \
			--moe_num_extend=8 \
			--moe_num_mixtures=8 \
			--distillation_features=True \
		   	--distillation_type=0 \
			--ensemble_w=0.2 \
			--noise_std=0.0 \
			--train=False \
			--run_once=True
	fi
done
