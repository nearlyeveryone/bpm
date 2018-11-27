import tensorflow as tf
import os
import subprocess
import glob
from time import time
import json
import utils
import bpm_estimator

tf.app.flags.DEFINE_string("mode", "s", "Either 's' for single, 'd' for distributed', 'g' for graph")
# tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
# tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
flags = tf.app.flags.FLAGS

with open('tf_config.json', 'r') as config:
    data = config.read()

# this config is for setting up distributed training
os.environ["TF_CONFIG"] = data


if flags.mode == 'd':
    pass
elif flags.mode == 's':
    #bpm_keras.train_sequential()
    #bpm_keras.train_seq_2()
    bpm_estimator.train_sequential()
    #bpm_estimator.predict_sequential('1543199066')
elif flags.mode == 'g':
    train_iterator = bpm_estimator.build_dataset_from_tfrecords(glob.glob(os.path.join(utils.working_dir, 'train_fragment_*.tfrecords')), 'train', num_repeat=1).make_one_shot_iterator()
    with tf.Session() as sess:
        while True:
            try:
                epoch_x, epoch_y = sess.run(train_iterator.get_next())
                for i in range(utils.batch_size):
                    utils.plot_stft(epoch_x['transformed_audio'][i], epoch_y[i])
                    print('...')

            except tf.errors.OutOfRangeError:
                break
elif flags.mode == 'v':
    bpm_estimator.visualize_data_transformations()
