GPU_ID=0
EVERY=1000
MODEL=DeepCnnDeepCombineChainModel
MODEL_DIR="../model/deep_cnn_deep_combine_chain"

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
                        --moe_num_mixtures=4 \
                        --deep_chain_layers=2 \
                        --deep_chain_relu_cells=256 \
			--deep_cnn_base_size=128 \
                        --label_loss=MultiTaskCrossEntropyLoss \
                        --multitask=True \
                        --support_type="label,label" \
                        --support_loss_percent=0.1 \
                        --batch_size=256 \
                        --model=$MODEL \
                        --run_once=True
        fi
done

