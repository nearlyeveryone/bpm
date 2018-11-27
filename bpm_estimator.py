import tensorflow as tf
import os
import time
import glob
import functools
import numpy as np

import utils


def serving_input_fn():
    feature_placeholders = {
        'raw_audio_data': tf.placeholder(tf.float32, [int(utils.audio_clip_len*utils.sample_rate)])
    }
    transformed_audio, _ = transform_data(feature_placeholders['raw_audio_data'], None)
    transformed_audio = tf.reshape(transformed_audio, [1, transformed_audio.shape[0], transformed_audio.shape[1]])
    zeros = tf.zeros([utils.batch_size-1, transformed_audio.shape[1], transformed_audio.shape[2]])
    transformed_audio = tf.concat([transformed_audio, zeros], 0)
    features = {
        'transformed_audio': transformed_audio
    }

    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


def train_input_fn():
    return build_dataset_from_tfrecords(
        glob.glob(os.path.join(utils.working_dir, 'train_fragment_*.tfrecords')), 'train', num_repeat=None)


def validation_input_fn():
    return build_dataset_from_tfrecords(
        glob.glob(os.path.join(utils.working_dir, 'val_fragment_*.tfrecords')), 'val', num_repeat=1)


def thresholding(inputs):
    # find the mean for each example in the batch
    mean_output = tf.reduce_mean(inputs, axis=1)

    # scale each mean based on a factor
    threshold_scalar = tf.Variable(utils.threshold_scalar, tf.float32)
    scaled_mean = tf.scalar_mul(threshold_scalar, mean_output)
    scaled_mean = tf.reshape(scaled_mean, [utils.batch_size])

    # setup matrix for
    min_thresh_for_max = tf.fill([utils.batch_size], 0.05)
    max_thresh_for_min = tf.fill([utils.batch_size], 0.15)   #0.4
    thresholds = tf.maximum(min_thresh_for_max, scaled_mean)
    thresholds = tf.minimum(max_thresh_for_min, thresholds)

    # zero values under the thresholds using bitmask
    thresholds = tf.reshape(thresholds, [128, 1, 1])

    threshold_mask = tf.cast(tf.greater(inputs, thresholds), tf.float32)
    thresholded_input = tf.multiply(inputs, threshold_mask)

    # peak picking
    # select beats by x[i-1] < x[i] > x[i+1] (local maximum)
    x_minus_1 = tf.cast(tf.greater(thresholded_input, tf.manip.roll(thresholded_input, shift=-1, axis=1)), tf.float32)
    x_plus_1 = tf.cast(tf.greater(thresholded_input, tf.manip.roll(thresholded_input, shift=1, axis=1)), tf.float32)
    output = tf.multiply(x_minus_1, x_plus_1)


    return output


def model_fn(features, labels, mode):
    # turn inputs into time-major instead of batch-
    inputs = tf.transpose(features['transformed_audio'], perm=[1, 0, 2])
    brnn = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=3,
                                          input_mode='linear_input',
                                          num_units=50,
                                          direction='bidirectional')
    brnn_output, hidden = brnn(inputs)
    # TODO: might want to normalize inputs because the sigmoid will fuck it?
    logits = tf.layers.dense(inputs=brnn_output, units=1, activation=tf.nn.sigmoid)
    logits = tf.transpose(logits, perm=[1, 0, 2])
    predictions = thresholding(logits)

    loss = None
    train_op = None
    metrics = None

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_pred=logits, y_true=labels))

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=utils.learning_rate)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
        # yay for metrics
        acc = tf.metrics.accuracy(predictions=predictions, labels=labels)
        pre = tf.metrics.precision(predictions=predictions, labels=labels)
        rec = tf.metrics.recall(predictions=predictions, labels=labels)

        metrics = {'accuracy': acc, 'precision': pre, 'recall': rec}
        tf.summary.scalar('accuracy', acc[1])
        tf.summary.scalar('precision', pre[1])
        tf.summary.scalar('recall', rec[1])

    result_dict = {
        'predictions': predictions,
        'probabilities': logits,
        'features': features['transformed_audio'],
        'brnn_output': tf.transpose(brnn_output, perm=[1, 0, 2])
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        predictions=result_dict,
        train_op=train_op,
        eval_metric_ops=metrics,
        export_outputs={'predictions': tf.estimator.export.PredictOutput(result_dict)}
    )


def train_sequential():
    tf.logging.set_verbosity(tf.logging.INFO)
    eval_interval = 300

    # strategy = tf.contrib.distribute.CollectiveAllReduceStrategy(num_gpus_per_worker=1)
    # config = tf.estimator.RunConfig(train_distribute=strategy, beval_distribute=strategy, model_dir=utils.nfs_dir)
    config = tf.estimator.RunConfig(model_dir=os.path.join(utils.working_dir, 'logs'), save_checkpoints_secs=eval_interval, keep_checkpoint_max=10)
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=config)
    # bidirectional_input
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=(utils.train_samples_to_use//utils.batch_size)*utils.num_epochs) #(utils.train_samples_to_use//utils.batch_size)*1) # utils.num_epochs

    # TODO: investigate exporter. it has to do with formatting data before it predicts?
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)

    eval_spec = tf.estimator.EvalSpec(input_fn=validation_input_fn,
                                      steps=None,
                                      exporters=exporter,
                                      start_delay_secs=60,
                                      throttle_secs=eval_interval)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    #estimator.export_saved_model(export_dir_base=os.path.join(utils.working_dir, 'logs/exports'),
    #                             serving_input_receiver_fn=serving_input_fn)
    predictoo = estimator.predict(input_fn=validation_input_fn)
    while True:
        predictions = next(predictoo)
        utils.plot_output(predictions['predictions'], predictions['probabilities'],
                          predictions['features'], predictions['brnn_output'])


def get_predict_dataset_from_np():
    import soundfile as sf
    audio_file = sf.SoundFile('/mnt/mirrored-storage/tf-workdir/extracted/209651_Afilia_Saga_-_S.M.L_(TV_size_ver.).osz/sml (tv size).mp3.wav')
    batch_x = audio_file.read(dtype='float32')
    batch_x = batch_x[0:int(utils.audio_clip_len * utils.sample_rate) + 1]
    #batch_x = np.reshape(batch_x, (1, batch_x.shape[0]))
    dataset = tf.data.Dataset.from_tensor_slices(batch_x)
    dataset.apply(transform_data)
    return dataset


def predict_sequential(export_dir):
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess,
                                   [tf.saved_model.tag_constants.SERVING],
                                   os.path.join(utils.working_dir, 'logs/exports/{}'.format(export_dir)))
        import soundfile as sf
        audio_file = sf.SoundFile(
            '/mnt/mirrored-storage/tf-workdir/extracted/209651_Afilia_Saga_-_S.M.L_(TV_size_ver.).osz/sml (tv size).mp3.wav')
        batch_x = audio_file.read(dtype='float32')
        batch_x = batch_x[0:int(utils.audio_clip_len * utils.sample_rate) + 1]
        batch_x = np.reshape(batch_x, (1, batch_x.shape[0]))
        input_list = batch_x.tolist()

        model_input = tf.train.Example(features=tf.train.Features(feature={'raw_audio_data': tf.train.Feature(float_list=tf.train.FloatList(value=input_list))}))

        predictor = tf.contrib.predictor.from_saved_model(os.path.join(utils.working_dir, 'logs/exports/{}'.format(export_dir)))
        model_input = model_input.SerializeToString()
        output_dict = predictor({"inputs": [model_input]})
        print(" prediction is ", output_dict['scores'])


def reshape_stft(stfts, num_mel_bins):
    magnitude_spectrograms = tf.abs(stfts)
    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value

    # scale frequency to mel scale and put into bins to reduce dimensionality
    lower_edge_hertz, upper_edge_hertz = 30.0, 17000.0

    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, utils.sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(
        magnitude_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # log scale the mel bins to better represent human loudness perception
    log_offset = 1e-6
    log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)

    # compute first order differential and concat. "It indicates a raise or reduction of the energy for each
    # frequency bin at a frame relative to its predecessor"
    first_order_diff = tf.abs(
        tf.subtract(log_mel_spectrograms, tf.manip.roll(log_mel_spectrograms, shift=1, axis=1)))
    mel_fod = tf.concat([log_mel_spectrograms, first_order_diff], 1)

    return mel_fod


def transform_data(audio, beatmap=None):
    # transform audio
    stfts = [tf.contrib.signal.stft(audio, frame_length=utils.frame_length, frame_step=utils.frame_step, fft_length=1024),
             tf.contrib.signal.stft(audio, frame_length=utils.frame_length, frame_step=utils.frame_step, fft_length=2048),
             tf.contrib.signal.stft(audio, frame_length=utils.frame_length, frame_step=utils.frame_step, fft_length=4096)]

    # one hot encode the 'binary' values for beats
    # beatmap = tf.one_hot(beatmap, utils.num_classes)
    if beatmap is not None:
        beatmap = tf.reshape(beatmap, [int(utils.timesteps), 1])
        beatmap = tf.cast(beatmap, tf.float32)

    # apply transforms to stfts, this includes the first order difference data
    mel_bins_base = utils.mel_bins_base
    log_mel_spectrograms = [reshape_stft(stfts[0], mel_bins_base),
                            reshape_stft(stfts[1], mel_bins_base*2),
                            reshape_stft(stfts[2], mel_bins_base*4)]

    # concat specs together
    stacked_log_mel_specs = tf.concat([log_mel_spectrograms[0], log_mel_spectrograms[1]], 1)
    stacked_log_mel_specs = tf.concat([stacked_log_mel_specs, log_mel_spectrograms[2]], 1)

    return stacked_log_mel_specs, beatmap


def parse_tfrecord(example_proto):
    keys_to_features = {'audio': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                        'beatmap': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)}
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    audio, beatmap = transform_data(parsed_features['audio'], parsed_features['beatmap'])
    return {'transformed_audio': audio}, beatmap


def build_dataset_from_tfrecords(records, tag, num_repeat):
    dataset = tf.data.TFRecordDataset(records)

    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        map_func=parse_tfrecord, batch_size=utils.batch_size, drop_remainder=True))
    # os.path.join(utils.working_dir, 'transformation_cache.dat')
    dataset = dataset.cache(os.path.join(utils.trans_cache_dir, '{}_transformation_cache.dat'.format(tag)))
    dataset = dataset.repeat(num_repeat)
    dataset = dataset.prefetch(4)

    return dataset


def parse_tfrecord_raw(example_proto):
    keys_to_features = {'audio': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                        'beatmap': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)}
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    return parsed_features['audio'], parsed_features['beatmap']


def visualize_data_transformations():
    records = glob.glob(os.path.join(utils.working_dir, 'train_fragment_*.tfrecords'))
    dataset = tf.data.TFRecordDataset(records)
    dataset = dataset.map(parse_tfrecord_raw)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=10)
    dataset = dataset.prefetch(2)
    it = dataset.make_one_shot_iterator()

    data_x = tf.placeholder(tf.float32, shape=(utils.sample_rate * utils.audio_clip_len,))
    data_y = tf.placeholder(tf.float32, shape=(utils.timesteps,))
    stfts = tf.contrib.signal.stft(data_x, frame_length=utils.frame_length, frame_step=utils.frame_step,
                                   fft_length=4096)
    power_stfts = tf.real(stfts * tf.conj(stfts))
    magnitude_spectrograms = tf.abs(stfts)
    power_magnitude_spectrograms = tf.abs(power_stfts)

    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value

    # scale frequency to mel scale and put into bins to reduce dimensionality
    lower_edge_hertz, upper_edge_hertz = 30.0, 17000.0
    num_mel_bins = utils.mel_bins_base * 4
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, utils.sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(
        magnitude_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # log scale the mel bins to better represent human loudness perception
    log_offset = 1e-6
    log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)

    # compute first order differential and concat. "It indicates a raise or reduction of the energy for each
    # frequency bin at a frame relative to its predecessor"
    first_order_diff = tf.abs(
        tf.subtract(log_mel_spectrograms, tf.manip.roll(log_mel_spectrograms, shift=1, axis=1)))
    mel_fod = tf.concat([log_mel_spectrograms, first_order_diff], 1)

    with tf.Session() as sess:
        while True:
            try:
                raw_x, raw_y = sess.run(it.get_next())
                np_stfts = sess.run(power_stfts, feed_dict={data_x: raw_x})
                np_magnitude_spectrograms = sess.run(power_magnitude_spectrograms, feed_dict={data_x: raw_x})
                np_mel_spectrograms = sess.run(mel_spectrograms, feed_dict={data_x: raw_x})
                np_log_mel_spectrograms = sess.run(log_mel_spectrograms, feed_dict={data_x: raw_x})
                np_mel_fod = sess.run(mel_fod, feed_dict={data_x: raw_x})

                utils.plot_signal_transforms(raw_x,
                                            np_stfts,
                                            np_magnitude_spectrograms,
                                            np_mel_spectrograms,
                                            np_log_mel_spectrograms,
                                            np_mel_fod)
                print('wank')

            except tf.errors.OutOfRangeError:
                break
