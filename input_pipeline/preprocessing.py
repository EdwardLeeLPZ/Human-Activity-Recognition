import gin
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d, laplace, gaussian_laplace


@gin.configurable
def write_tf_record_files(data_dir, window_size, window_shift, num_categories=13):
    """build TFRecord file of HAPT dataset

    Parameters:
        data_dir (string): original path directory where the data is stored
        windows_size (int): length of sequence window (Default is 250)
        windows_shift (int): overlap length of sequence windows (Default is 125)
        num_categories (int): number of label category (must be 12 or 13, when use 12, then remove the data with label 0)
    """

    # define train, validation and test experiment ranges
    train_experiment_ids = range(1, 44)
    validation_experiment_ids = range(56, 62)
    test_experiment_ids = range(44, 56)
    total_experiment_ids = range(1, 62)

    # seperate original labels according to experiment numbers
    separated_labels = [None] * 61
    original_labels = pd.read_csv(data_dir + "/HAPT_dataset/RawData/labels.txt", sep=' ', header=None)
    for experiment_id in total_experiment_ids:
        experiment_labels = []
        for label in original_labels.values:
            if experiment_id == label[0]:
                experiment_labels.append(label)
        separated_labels[experiment_id - 1] = experiment_labels

    # get lists of experiment files
    acc_files = [file for file in os.listdir(data_dir + "/HAPT_dataset/RawData/") if file.startswith('acc')]
    gyro_files = [file for file in os.listdir(data_dir + "/HAPT_dataset/RawData/") if file.startswith('gyro')]
    acc_files.sort()
    gyro_files.sort()

    # read data sequences and labels from original files
    train_features, validation_features, test_features = [], [], []
    train_labels, validation_labels, test_labels = [], [], []
    for acc_file, gyro_file, experiment_id in zip(acc_files, gyro_files, total_experiment_ids):
        acc_data = pd.read_csv(data_dir + "/HAPT_dataset/RawData/" + acc_file, sep=' ', names=['x', 'y', 'z'])
        gyro_data = pd.read_csv(data_dir + "/HAPT_dataset/RawData/" + gyro_file, sep=' ', names=['x', 'y', 'z'])
        assert len(acc_data['x']) == len(gyro_data['x'])
        # normalize data and reshape into six channels by using Z-score
        sequence_data = np.asarray(
            [stats.zscore(acc_data['x']), stats.zscore(acc_data['y']), stats.zscore(acc_data['z']),
             stats.zscore(gyro_data['x']), stats.zscore(gyro_data['y']), stats.zscore(gyro_data['z'])]).transpose()
        sequence_length = len(acc_data['x'])
        # initiate labels and change labels according to slices in label file
        sequence_labels = [0] * sequence_length
        for label in separated_labels[experiment_id - 1]:
            sequence_labels[label[3]: label[4] + 1] = [label[2]] * (label[4] + 1 - label[3])
        if experiment_id in train_experiment_ids:
            train_features.append(sequence_data)
            train_labels.append(sequence_labels)
        elif experiment_id in validation_experiment_ids:
            validation_features.append(sequence_data)
            validation_labels.append(sequence_labels)
        elif experiment_id in test_experiment_ids:
            test_features.append(sequence_data)
            test_labels.append(sequence_labels)
        else:
            raise ValueError
    train_features = np.concatenate(train_features)
    validation_features = np.concatenate(validation_features)
    test_features = np.concatenate(test_features)
    train_labels = np.concatenate(train_labels)
    validation_labels = np.concatenate(validation_labels)
    test_labels = np.concatenate(test_labels)

    # preprocessing (filtering) the data
    train_features = preprocessing(train_features)
    validation_features = preprocessing(validation_features)
    test_features = preprocessing(test_features)

    # delete data with label 0 in datasets and change 13 classes to 12 classes
    def delete_zero_label(features, labels):
        """delete data with label 0"""
        index = np.argwhere(labels == [0])
        features = np.delete(features, index, axis=0)
        labels = np.delete(labels, index, axis=0)
        labels -= 1
        return features, labels

    if num_categories == 12:
        train_features, train_labels = delete_zero_label(train_features, train_labels)
        validation_features, validation_labels = delete_zero_label(validation_features, validation_labels)
        test_features, test_labels = delete_zero_label(test_features, test_labels)

    # generate datasets and split them into sliding windows
    ds_train = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    ds_train = ds_train.window(size=window_size, shift=window_shift, drop_remainder=True)
    ds_val = tf.data.Dataset.from_tensor_slices((validation_features, validation_labels))
    ds_val = ds_val.window(size=window_size, shift=window_shift, drop_remainder=True)
    ds_test = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
    ds_test = ds_test.window(size=window_size, shift=window_size, drop_remainder=True)
    ds_train = ds_train.flat_map(
        lambda feature_window, label_window: tf.data.Dataset.zip((feature_window, label_window))).batch(window_size, drop_remainder=True)
    ds_val = ds_val.flat_map(
        lambda feature_window, label_window: tf.data.Dataset.zip((feature_window, label_window))).batch(window_size, drop_remainder=True)
    ds_test = ds_test.flat_map(
        lambda feature_window, label_window: tf.data.Dataset.zip((feature_window, label_window))).batch(window_size, drop_remainder=True)

    # define features
    def _bytes_feature(value):
        """returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def window_example(feature_window, label_window):
        """change the sequence string into an example"""
        feature_window = tf.io.serialize_tensor(feature_window).numpy()
        label_window = tf.io.serialize_tensor(label_window).numpy()
        feature = {'window_sequence': _bytes_feature(feature_window),
                   'label': _bytes_feature(label_window)}
        return tf.train.Example(features=tf.train.Features(feature=feature))

    # create TFRecord directories and files
    dir_name = 'window_size_' + str(window_size) + '_window_shift_' + str(window_shift)
    if not os.path.exists(os.path.join(os.getcwd(), 'TFRecord', dir_name)):
        os.makedirs(os.path.join(os.getcwd(), 'TFRecord', dir_name))
    with tf.io.TFRecordWriter("./TFRecord/" + dir_name + "/train.tfrecord") as writer:
        for feature_window, label_window in ds_train:
            tf_example = window_example(feature_window, label_window)
            writer.write(tf_example.SerializeToString())
    with tf.io.TFRecordWriter("./TFRecord/" + dir_name + "/validation.tfrecord") as writer:
        for feature_window, label_window in ds_val:
            tf_example = window_example(feature_window, label_window)
            writer.write(tf_example.SerializeToString())
    with tf.io.TFRecordWriter("./TFRecord/" + dir_name + "/test.tfrecord") as writer:
        for feature_window, label_window in ds_test:
            tf_example = window_example(feature_window, label_window)
            writer.write(tf_example.SerializeToString())


def read_tfrecord(window_size, window_shift, num_categories):
    """read the original dataset from TFRecord

    Parameters:
        windows_size (int): length of sequence window (Default is 250)
        windows_shift (int): overlap length of sequence windows (Default is 125)
        num_categories (int): number of label category (must be 12 or 13, when use 12, then remove the data with label 0)
    """

    # import the original training validation and test datasets from TFRecord files
    dir_name = 'window_size_' + str(window_size) + '_window_shift_' + str(window_shift)
    ds_train = tf.data.TFRecordDataset('./TFRecord/' + dir_name + '/train.tfrecord')
    ds_val = tf.data.TFRecordDataset('./TFRecord/' + dir_name + '/validation.tfrecord')
    ds_test = tf.data.TFRecordDataset('./TFRecord/' + dir_name + '/test.tfrecord')
    ds_info = {'window_sequence': tf.io.FixedLenFeature([], tf.string),
               'label': tf.io.FixedLenFeature([], tf.string)}

    # parse data from bytes format into original sequence format
    def _parse_example(window_example):
        """parse data"""
        temp = tf.io.parse_single_example(window_example, ds_info)
        feature_window = tf.io.parse_tensor(temp['window_sequence'], tf.float64)
        label_window = tf.io.parse_tensor(temp['label'], tf.int64)
        return feature_window, label_window

    ds_train = ds_train.map(_parse_example)
    ds_val = ds_val.map(_parse_example)
    ds_test = ds_test.map(_parse_example)

    def drawing(dataset, name):
        """visualize the original data and their distribution"""
        acc_x, acc_y, acc_z = [], [], []
        gyro_x, gyro_y, gyro_z = [], [], []
        labels = []
        for idx, (window_sequence, label) in enumerate(dataset):
            window_sequence = window_sequence.numpy()
            labels.append(label.numpy())
            acc_x.append(window_sequence[:, 0]), acc_y.append(window_sequence[:, 1]), acc_z.append(window_sequence[:, 2])
            gyro_x.append(window_sequence[:, 3]), gyro_y.append(window_sequence[:, 4]), gyro_z.append(window_sequence[:, 5])
            if idx >= 50:
                break
        acc_x, acc_y, acc_z = np.concatenate(acc_x), np.concatenate(acc_y), np.concatenate(acc_z)
        gyro_x, gyro_y, gyro_z = np.concatenate(gyro_x), np.concatenate(gyro_y), np.concatenate(gyro_z)
        labels = np.concatenate(labels)

        def labeling(labels, num_categories):
            """define the color map corresponding to each label"""
            if num_categories == 13:
                label_color = ['white', 'dimgrey', 'darkorange', 'limegreen', 'royalblue', 'lightcoral', 'gold',
                               'aquamarine', 'mediumslateblue', 'saddlebrown', 'chartreuse', 'skyblue', 'violet']
            else:
                label_color = ['dimgrey', 'darkorange', 'limegreen', 'royalblue', 'lightcoral', 'gold',
                               'aquamarine', 'mediumslateblue', 'saddlebrown', 'chartreuse', 'skyblue', 'violet']
            start = 0
            for i in range(1, int(labels.size)):
                if labels[i] != labels[i - 1]:
                    end = i - 1
                    plt.axvspan(start, end, facecolor=label_color[labels[i - 1]], alpha=0.5)
                    start = i
            plt.axvspan(start, int(labels.size) - 1, facecolor=label_color[labels[-1]], alpha=0.5)

        # intercept sequence of a certain length
        view_size = 13000
        acc_x, acc_y, acc_z = acc_x[:view_size], acc_y[:view_size], acc_z[:view_size]
        gyro_x, gyro_y, gyro_z = gyro_x[:view_size], gyro_y[:view_size], gyro_z[:view_size]
        labels = labels[:view_size]

        # plot the sequence and label
        plt.figure(figsize=(9, 6), dpi=150)
        ax1 = plt.subplot(2, 1, 1)
        plt.title('ACCELEROMETER DATA', fontdict={'weight': 'normal', 'size': 'x-large'})
        plt.tick_params(labelsize='xx-small')
        ax1.set_xlabel("TIME SEQUENCE")
        ax1.set_ylabel("NORMALIZED ACCELERATION")
        plt.plot(acc_x, label='acc_x', linewidth=1)
        plt.plot(acc_y, label='acc_y', linewidth=1)
        plt.plot(acc_z, label='acc_z', linewidth=1)
        labeling(labels, num_categories)
        ax2 = plt.subplot(2, 1, 2)
        plt.title('GYROSCOPE DATA', fontdict={'weight': 'normal', 'size': 'x-large'})
        plt.tick_params(labelsize='xx-small')
        ax2.set_xlabel("TIME SEQUENCE")
        ax2.set_ylabel("NORMALIZED AUGULAR ACCELERATION")
        plt.plot(gyro_x, label='gyro_x', linewidth=1)
        plt.plot(gyro_y, label='gyro_y', linewidth=1)
        plt.plot(gyro_z, label='gyro_z', linewidth=1)
        labeling(labels, num_categories)
        plt.savefig('./' + name + '.png', dpi=150)

        plt.show()

    drawing(ds_train, 'ds_train')
    drawing(ds_val, 'ds_val')
    drawing(ds_test, 'ds_test')


@gin.configurable
def preprocessing(features, method):
    """preprocess the sequence data

    Parameters:
        features (numpy array): original data sequence
        method (dictionary): preprocessing methods (filters) and sizes of filters

    Returns:
         features (numpy array): data sequence after preprocessing
    """
    for (key, value) in method:
        if key == 'gaussian':
            for i in range(6):
                features[:, i] = gaussian_filter1d(features[:, i], sigma=value, axis=0)
        elif key == 'median':
            for i in range(6):
                features[:, i] = median_filter(features[:, i], size=value)
        elif key == 'uniform':
            for i in range(6):
                features[:, i] = uniform_filter1d(features[:, i], size=value, axis=0)
        elif key == 'laplace':
            for i in range(6):
                features[:, i] = laplace(features[:, i])
        elif key == 'gaussian_laplace':
            for i in range(6):
                features[:, i] = gaussian_laplace(features[:, i], sigma=value)
    return features