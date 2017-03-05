import sys
import tensorflow as tf

def read_and_decode_single_example(filepattern, additional_features, feature_sizes):
	# first construct a queue containing a list of filenames.
	# this lets a user split up there dataset in multiple files to keep
	# size down
	filelist = tf.gfile.Glob(filepattern)
	if not filelist:
		print >> sys.stderr, "no files found in", filepattern
	filename_queue = tf.train.string_input_producer(filelist, num_epochs=1, shuffle = False)

	# Unlike the TFRecordWriter, the TFRecordReader is symbolic
	reader = tf.TFRecordReader()

	# One can read a single serialized example from a filename
	# serialized_example is a Tensor of type string.
	_, serialized_example = reader.read(filename_queue)

	# The serialized example is converted back to actual values.
	# One needs to describe the format of the objects to be returned
	feature_map = {"video_id": tf.FixedLenFeature([], tf.string), 
		"labels": tf.VarLenFeature(tf.int64)}

	for feature_name in additional_features:
		feature_map[feature_name] = tf.FixedLenFeature([feature_sizes[feature_name]], tf.float32)

	features = tf.parse_single_example(serialized_example, feature_map)
	concat_feature = tf.concat([features[name] for name in additional_features], axis = 0)
	return features["video_id"], features["labels"].values, concat_feature

if __name__ == "__main__":

	if len(sys.argv) > 3:
		filepattern = sys.argv[1]
		additional_features = [s.strip() for s in sys.argv[2].split(",")]
		feature_sizes = [int(s) for s in sys.argv[3].split(",")]
		feature_sizes = dict(zip(additional_features, feature_sizes))
	else:
		sys.exit("Usage: python %s filepattern additional_features feature_sizes")

	with tf.Session() as sess:
		video_id, label, feature = read_and_decode_single_example(filepattern, additional_features, feature_sizes)
		global_init = tf.variables_initializer(tf.global_variables(), name = "global_init")
		local_init = tf.variables_initializer(tf.local_variables(), name = "local_init")
		sess.run(global_init)
		sess.run(local_init)
		tf.train.start_queue_runners(sess = sess)
		i = 0
		while True:
			v_id, l_val, f_val = sess.run([video_id, label, feature])
			l_list = l_val.flatten().tolist()
			f_list = f_val.flatten().tolist()
			l_len = len(l_list)
			f_len = len(f_list)
			string = "%s\t%d %s %d %s" % (v_id, l_len, " ".join(map(str, l_list)), f_len, " ".join(map(str, f_list)))
			print string

