from __future__ import print_function

import tensorflow as tf
from tensorflow import app
from tensorflow import flags
import numpy as np
import utils

flags.DEFINE_string("src_path_1", "/home/share/zhangt/mywork/youtube-8m-master/data/video/train_my/trainL_.tfrecord", "")
flags.DEFINE_string("src_path_2", "/home/share/Youtube-8M/data/video/validate/*.tfrecord", "")

def get_frame_input_feature(input_file):
    features = []
    record_iterator = tf.python_io.tf_record_iterator(path=input_file)
    for i, string_record in enumerate(record_iterator):
        example = tf.train.SequenceExample()
        example.ParseFromString(string_record)

        # traverse the Example format to get data
        video_id = example.context.feature['video_id'].bytes_list.value[0]
        label = example.context.feature['labels'].int64_list.value[:]
        rgbs = []
        audios = []
        rgb_feature = example.feature_lists.feature_list['rgb'].feature
        for i in range(len(rgb_feature)):
            rgb = np.fromstring(rgb_feature[i].bytes_list.value[0], dtype=np.uint8).astype(np.float32)
            rgb = utils.Dequantize(rgb, 2, -2)
            rgbs.append(rgb)
        audio_feature = example.feature_lists.feature_list['audio'].feature
        for i in range(len(audio_feature)):
            audio = np.fromstring(audio_feature[i].bytes_list.value[0], dtype=np.uint8).astype(np.float32)
            audio = utils.Dequantize(audio, 2, -2)
            audios.append(audio)
        rgbs = np.array(rgbs)
        audios = np.array(audios)
        features.append((video_id, label, rgbs, audios))
    return features

def get_video_input_feature(input_file):
    features = []
    record_iterator = tf.python_io.tf_record_iterator(path=input_file)
    for i, string_record in enumerate(record_iterator):
        example = tf.train.Example()
        example.ParseFromString(string_record)

        # traverse the Example format to get data
        video_id = example.features.feature['video_id'].bytes_list.value[0]
        label = example.features.feature['labels'].int64_list.value[:]
        mean_rgb = example.features.feature['mean_rgb'].float_list.value[:]
        mean_audio = example.features.feature['mean_audio'].float_list.value[:]
        std_rgb = example.features.feature['std_rgb'].float_list.value[:]
        std_audio = example.features.feature['std_audio'].float_list.value[:]
        #features.append(label)
        features.append((video_id, label, mean_rgb, mean_audio, std_rgb, std_audio))
    return features

def get_output_feature(video_id, labels, features, feature_names):
    feature_maps = {'video_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[video_id])),
                    'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))}
    for feature_index in range(len(feature_names)):
        feature_maps[feature_names[feature_index]] = tf.train.Feature(
            float_list=tf.train.FloatList(value=features[feature_index]))
    example = tf.train.Example(features=tf.train.Features(feature=feature_maps))
    return example

def main():
    files = tf.gfile.Glob(flags.FLAGS.src_path_1)
    labels_uni = np.zeros([4716,1])
    labels_matrix = np.zeros([4716,4716])
    for file in files:
        labels_all = get_video_input_feature(file)
        print(len(labels_all[0][2]),len(labels_all[0][3]),len(labels_all[0][4]),len(labels_all[0][5]))
        """
        for labels in labels_all:
            for i in range(len(labels)):
                labels_uni[labels[i]] += 1
                for j in range(len(labels)):
                    labels_matrix[labels[i],labels[j]] += 1
    labels_matrix = labels_matrix/labels_uni
    labels_matrix = labels_matrix/(np.sum(labels_matrix,axis=0)-1.0)
    for i in range(4716):
        labels_matrix[i,i] = 1.0
    np.savetxt('labels_uni.out', labels_uni, delimiter=',')
    np.savetxt('labels_matrix.out', labels_matrix, delimiter=',')"""

if __name__=='__main__':
    main()
