#!/bin/bash
OUT_PATH_1="/home/share/zhangt/mywork/youtube-8m-master/data/video/train_my/"
IN_PATH_1="/Youtube-8M/data/frame/train/"
OUT_PATH_2="/home/share/zhangt/mywork/youtube-8m-master/data/video/valid_my/"
IN_PATH_2="/Youtube-8M/data/frame/validate/"
OUT_PATH_3="/home/share/zhangt/mywork/youtube-8m-master/data/video/test_my/"
IN_PATH_3="/Youtube-8M/data/frame/test/"
for file in `find $IN_PATH_1 -name "*.tfrecord"`
    do
        echo "echo 'Testing $file' && CUDA_VISIBLE_DEVICES='' python YM_readframefeature.py --src_path $file --des_path ${file/$IN_PATH_1/$OUT_PATH_1}"
    done
for file in `find $IN_PATH_2 -name "*.tfrecord"`
    do
        echo "echo 'Testing $file' && CUDA_VISIBLE_DEVICES='' python YM_readframefeature.py --src_path $file --des_path ${file/$IN_PATH_2/$OUT_PATH_2}"
    done
for file in `find $IN_PATH_3 -name "*.tfrecord"`
    do
        echo "echo 'Testing $file' && CUDA_VISIBLE_DEVICES='' python YM_readframefeature.py --src_path $file --des_path ${file/$IN_PATH_3/$OUT_PATH_3}"
    done
