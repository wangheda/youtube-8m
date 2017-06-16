GPU_ID="1"
EVERY=500
MODEL=ChainSupportReluMoeModel 
MODEL_DIR="../model/layer_chain_moe8_vert"

start=$1
DIR="$(pwd)"

for checkpoint in $(cd $MODEL_DIR && python ${DIR}/training_utils/select.py $EVERY); do
        echo $checkpoint;
        if [ $checkpoint -gt $start ]; then
                echo $checkpoint;
                CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
                        --train_dir="$MODEL_DIR" \
                        --model_checkpoint_path="${MODEL_DIR}/model.ckpt-${checkpoint}" \
                        --eval_data_pattern="/Youtube-8M/data/layer/validate/validate*0.tfrecord" \
                        --frame_features=False \
                        --feature_names="layer" \
                        --feature_sizes="2048" \
                        --batch_size=128 \
                        --num_supports=25 \
                        --support_type="vertical" \
                        --model=$MODEL \
                        --moe_num_mixtures=8 \
                        --run_once=True
        fi
done

