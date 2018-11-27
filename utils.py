import os
from matplotlib import pyplot as plt
import numpy as np
import math

# directory settings

working_dir = '/mnt/mirrored-storage/tf-workdir'
trans_cache_dir = '/mnt/nvme-storage/tf'
nfs_dir = '/tmp/tf'
db_path = 'sqlite:///' + os.path.join(working_dir, 'beatmap_metadata.db')
zipped_beatmaps_dir = os.path.join(working_dir, 'beatmaps')
beatmap_extract_dir = os.path.join(working_dir, 'extracted')
checkpoint_path = os.path.join(working_dir, 'beat-prediction-model.{epoch:02d}-{val_loss:.2f}.hdf5')
latest_path = os.path.join(working_dir, 'beat-prediction-latest.hdf5')

# load number of samples in the tfrecords
try:
    file = open(os.path.join(working_dir, 'tfrecord_data.txt'))
    lines = [line.rstrip('\n') for line in file]
    train_samples_to_use = int(lines[0])
    val_samples_to_use = int(lines[1])
    test_samples_to_use = int(lines[2])
except OSError:
    pass

# audio params
sample_rate = 44100
frame_rate = 50
frame_step = math.ceil(sample_rate/frame_rate)
frame_length = frame_step*2
audio_clip_len = 26.46  # 44100/10000 * 6

mel_bins_base = 27

# training params
learning_rate = 0.00005
num_epochs = 20

# network parameters
num_input = mel_bins_base * 14
timesteps = audio_clip_len * frame_rate - 1
batch_size = 128
num_classes = 2
threshold_scalar = 2.0


def plot_output(predictions, probabilities, features, brnn_output):
    fig = plt.figure(1)
    fig.suptitle('predictions and probabilities')

    plt.subplot(411)
    plt.plot(predictions)
    plt.xlabel('timestep')
    plt.ylabel('predictions')

    plt.subplot(412)
    plt.plot(probabilities)
    plt.xlabel('timestep')
    plt.ylabel('probability')

    plt.subplot(413)
    _plot_rotate_stft(features)
    plt.xlabel('input features at timestep')

    plt.subplot(414)
    _plot_rotate_stft(brnn_output)
    plt.xlabel('output of brnn at timestep')

    plt.show()
    print(np.mean(probabilities))


def plot_stft(stft, bmap):
    stft = np.rot90(stft, axes=(0, 1))
    plt.figure(1)
    plt.subplot(311)
    plt.pcolormesh(stft, cmap=plt.get_cmap('viridis'))

    plt.subplot(312)
    plt.plot(bmap)

    ax1 = plt.subplot(313)
    ax1.pcolormesh(stft, cmap=plt.get_cmap('viridis'), zorder=1)

    ax2 = ax1.twinx()
    ax2.plot(bmap, zorder=2, color='red')

    plt.show()


def _plot_rotate_stft(stft, clip=False):
    stft = np.rot90(stft, axes=(0, 1))
    if clip:
        stft = np.clip(stft, 1, 500)
    plt.pcolormesh(stft, cmap=plt.get_cmap('viridis'))


def plot_signal_transforms(raw_x,
                           np_stfts,
                           np_magnitude_spectrograms,
                           np_mel_spectrograms,
                           np_log_mel_spectrograms,
                           np_mel_fod):
    plt.figure(1)
    plt.subplot(611, axisbg='black')

    plt.plot(raw_x, color='white')
    plt.xlabel('raw audio signal')

    plt.subplot(612)
    _plot_rotate_stft(np_stfts, True)
    plt.xlabel('stfts')

    plt.subplot(613)
    _plot_rotate_stft(np_magnitude_spectrograms, True)
    plt.xlabel('magnitude stfts')

    plt.subplot(614)
    _plot_rotate_stft(np_mel_spectrograms)
    plt.xlabel('magnitude mel stfts')

    plt.subplot(615)
    _plot_rotate_stft(np_log_mel_spectrograms)
    plt.xlabel('log magnitude mel stfts')

    plt.subplot(616)
    _plot_rotate_stft(np_mel_fod)
    plt.xlabel('magnitude mel stfts + first order differences')

    plt.show()
