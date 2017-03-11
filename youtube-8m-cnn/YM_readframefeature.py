from __future__ import print_function

import tensorflow as tf
from tensorflow import app
from tensorflow import flags
import numpy as np
import utils

flags.DEFINE_string("src_path", "./train-A.tfrecord", "")
flags.DEFINE_string("des_path", "./train-A-video.tfrecord", "")

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

    features = get_frame_input_feature(flags.FLAGS.src_path)
    writer = tf.python_io.TFRecordWriter(flags.FLAGS.des_path)

    #mean_features = np.loadtxt('frame_means.out', delimiter=',')
    #std_features = np.loadtxt('frame_stds.out', delimiter=',')

    for i, feature in enumerate(features):
        """
        feature_1 = (np.mean(feature[2], axis=0)-mean_features[0])/std_features[0]
        feature_2 = (np.mean(feature[3], axis=0)-mean_features[1])/std_features[1]
        feature_3 = (np.std(feature[2], axis=0)-mean_features[2])/std_features[2]
        feature_4 = (np.std(feature[3], axis=0)-mean_features[3])/std_features[3]"""
        feature_1 = np.mean(feature[2], axis=0)
        feature_2 = np.mean(feature[3], axis=0)
        feature_3 = np.std(feature[2], axis=0)
        feature_4 = np.std(feature[3], axis=0)
        example = get_output_feature(feature[0], feature[1], [feature_1, feature_2, feature_3, feature_4],
                                     ['mean_rgb','mean_audio','std_rgb','std_audio'])
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()
    """
    features = get_video_input_feature(flags.FLAGS.des_path)
    features_all = []
    for i, feature in enumerate(features):
        feature_1 = np.reshape(np.array(feature[2]),[len(feature[2]),1])
        feature_2 = np.reshape(np.array(feature[3]),[len(feature[3]),1])
        feature_3 = np.reshape(np.array(feature[4]),[len(feature[4]),1])
        feature_4 = np.reshape(np.array(feature[5]),[len(feature[5]),1])
        if features_all:
            feature_1 = np.concatenate([features_all[0],feature_1],axis=1)
            feature_2 = np.concatenate([features_all[1],feature_2],axis=1)
            feature_3 = np.concatenate([features_all[2],feature_3],axis=1)
            feature_4 = np.concatenate([features_all[3],feature_4],axis=1)
            features_all = [feature_1,feature_2,feature_3,feature_4]
        else:
            features_all = [feature_1,feature_2,feature_3,feature_4]
    mean_features = []
    std_features = []
    for i in range(len(features_all)):
        mean_features.append(np.mean(features_all[i],axis=1))
        std_features.append(np.std(features_all[i],axis=1))
    print(mean_features,std_features)"""

if __name__=='__main__':
    main()
