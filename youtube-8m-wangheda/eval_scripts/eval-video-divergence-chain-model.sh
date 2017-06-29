GPU_ID=0
EVERY=200
MODEL=MultiTaskDivergenceDeepCombineChainModel
MODEL_DIR="../model/video_divergence_deep_combine_chain"

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
            --batch_size=128 \
            --model=$MODEL \
            --lstm_cells="1024,128" \
            --divergence_model_count=8 \
            --deep_chain_layers=2 \
            --deep_chain_relu_cells=128 \
            --moe_num_mixtures=4 \
            --multitask=True \
            --label_loss=MultiTaskDivergenceCrossEntropyLoss \
            --support_loss_percent=0.025 \
            --num_readers=1 \
            --run_once=True
    fi
done

