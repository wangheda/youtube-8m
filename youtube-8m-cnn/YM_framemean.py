from __future__ import print_function

import tensorflow as tf
from tensorflow import app
from tensorflow import flags
import numpy as np
import os, fnmatch
import threading
import Queue
import utils

flags.DEFINE_string("src_path", "/Youtube-8M/data/frame/train", "")
flags.DEFINE_string("des_path", "./mean_std.tfrecord", "")

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

def get_output_files(features, feature_names):
    feature_maps = {}
    for feature_index in range(len(feature_names)):
        feature_maps[feature_names[feature_index]] = tf.train.Feature(
            float_list=tf.train.FloatList(value=features[feature_index]))
    example = tf.train.Example(features=tf.train.Features(feature=feature_maps))
    return example

def get_output_feature(video_id, labels, features, feature_names):
    feature_maps = {'video_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[video_id])),
                    'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))}
    for feature_index in range(len(feature_names)):
        feature_maps[feature_names[feature_index]] = tf.train.Feature(
            float_list=tf.train.FloatList(value=features[feature_index]))
    example = tf.train.Example(features=tf.train.Features(feature=feature_maps))
    return example

def read_batch_files(q,files):
    mean_features = []
    std_features = []
    mean_all = []
    std_all = []
    for file in files:
        print('processing '+file)
        features = get_frame_input_feature(flags.FLAGS.src_path+'/'+file)
        features_all = []

        for i, feature in enumerate(features):
            feature_1 = np.reshape(np.mean(feature[2], axis=0),[feature[2].shape[1],1])
            feature_2 = np.reshape(np.mean(feature[3], axis=0),[feature[3].shape[1],1])
            feature_3 = np.reshape(np.std(feature[2], axis=0),[feature[2].shape[1],1])
            feature_4 = np.reshape(np.std(feature[3], axis=0),[feature[3].shape[1],1])
            if features_all:
                feature_1 = np.concatenate([features_all[0],feature_1],axis=1)
                feature_2 = np.concatenate([features_all[1],feature_2],axis=1)
                feature_3 = np.concatenate([features_all[2],feature_3],axis=1)
                feature_4 = np.concatenate([features_all[3],feature_4],axis=1)
                features_all = [feature_1,feature_2,feature_3,feature_4]
            else:
                features_all = [feature_1,feature_2,feature_3,feature_4]

        mean_1 = []
        std_1 = []
        mean_all_1 = []
        std_all_1 = []
        for i in range(4):
            mean_1.append(np.reshape(np.mean(features_all[i], axis=1),[features_all[i].shape[0],1]))
            std_1.append(np.reshape(np.std(features_all[i], axis=1),[features_all[i].shape[0],1]))
            mean_all_1.append(np.reshape(np.mean(features_all[i]),[1,1]))
            std_all_1.append(np.reshape(np.std(features_all[i]),[1,1]))
        if mean_features:
            for i in range(4):
                mean_features[i] = np.concatenate([mean_features[i],mean_1[i]],axis=1)
                std_features[i] = np.concatenate([std_features[i],std_1[i]],axis=1)
                mean_all[i] = np.concatenate([mean_all[i],mean_all_1[i]],axis=1)
                std_all[i] = np.concatenate([std_all[i],std_all_1[i]],axis=1)
        else:
            for i in range(4):
                mean_features.append(mean_1[i])
                std_features.append(std_1[i])
                mean_all.append(mean_all_1[i])
                std_all.append(std_all_1[i])
    q.put([mean_features,std_features,mean_all,std_all])

class myThread(threading.Thread):
    def __init__(self, threadID, name, files, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.files = files
        self.q = q
    def run(self):
        print("Starting " + self.name)
        read_batch_files(self.q, self.files)


def main():

    files = fnmatch.filter(os.listdir(flags.FLAGS.src_path), '*.tfrecord')
    threads =[]
    thread_num = 10
    fstep = int(len(files)/thread_num)+1
    q = Queue.Queue()
    for i in range(thread_num):
        thread = myThread(i, "Thread-%d" % i, files[fstep*i:fstep*(i+1)], q)
        threads.append(thread)
        # Wait for all threads to complete
    print(threads)
    mean_features = []
    std_features = []
    mean_all = []
    std_all = []
    #Start all threads in thread pool
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    while not q.empty():
        response = q.get()
        mean_features.extend(response[0])
        std_features.extend(response[1])
        mean_all.extend(response[2])
        std_all.extend(response[3])
    for i in range(4):
        mean_features[i] = np.mean(mean_features[i],axis=1)
        std_features[i] = np.mean(std_features[i],axis=1)
        mean_all[i] = np.mean(mean_all[i],axis=1)
        std_all[i] = np.mean(std_all[i],axis=1)

    mean_all = np.array(mean_all)
    std_all = np.array(std_all)
    mean_features.extend(std_features)
    mean_features.append(mean_all)
    mean_features.append(std_all)


    writer = tf.python_io.TFRecordWriter(flags.FLAGS.des_path)
    example = get_output_files(mean_features,['mean_rgb_mean','mean_audio_mean','std_rgb_mean','std_audio_mean',
                                  'mean_rgb_std','mean_audio_std','std_rgb_std','std_audio_std','mean_4','std_4'])
    serialized = example.SerializeToString()
    writer.write(serialized)
    writer.close()

if __name__=='__main__':
    main()
