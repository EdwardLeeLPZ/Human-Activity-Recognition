from input_pipeline.preprocessing import *


@gin.configurable
def load(num_categories, name, data_dir, window_size, window_shift, batch_size, buffer_size, tfrecord_exist):
    """Loads data from files

    Parameters:
        num_categories (int): number of label category (must be 12 or 13, when use 12, then remove the data with label 0)
        name (string): name of the dataset (name list: 'HAPT', )
        data_dir (string): original path directory where the data is stored
        windows_size (int): length of sequence window (Default is 250)
        windows_shift (int): overlap length of sequence windows (Default is 125)
        batch_size (int): size of dataset batch
        buffer_size (int):  size of the shuffle buffer
        tfrecord_exist (bool): whether TFRecord exists or not (Default is False)

    Returns:
        ds_train (tf.data.Dataset): training set
        ds_val (tf.data.Dataset): validation set
        ds_test (tf.data.Dataset): test set
        ds_info (dictionary): information and structure of dataset
    """

    if name == 'HAPT':
        dir_name = 'window_size_' + str(window_size) + '_window_shift_' + str(window_shift)

        # check if there are already some existed TFRecord files. If not, then build the new TFRecord file
        if not tfrecord_exist:
            write_tf_record_files(data_dir, window_size=window_size, window_shift=window_shift, num_categories=num_categories)

        # visualize the original data distribution (optional)
        # read_tfrecord(window_size=window_size, window_shift=window_shift, num_categories=num_categories)

        # import the original training validation and test datasets from TFRecord files
        ds_train = tf.data.TFRecordDataset('./TFRecord/' + dir_name + '/train.tfrecord')
        ds_val = tf.data.TFRecordDataset('./TFRecord/' + dir_name + '/validation.tfrecord')
        ds_test = tf.data.TFRecordDataset('./TFRecord/' + dir_name + '/test.tfrecord')
        ds_info = {'window_sequence': tf.io.FixedLenFeature([], tf.string),
                   'label': tf.io.FixedLenFeature([], tf.string)}

        # parse data from bytes format into original sequence format
        def _parse_example(window_example):
            temp = tf.io.parse_single_example(window_example, ds_info)
            feature_window = tf.io.parse_tensor(temp['window_sequence'], tf.float64)
            label_window = tf.io.parse_tensor(temp['label'], tf.int64)
            return feature_window, label_window

        ds_train = ds_train.map(_parse_example)
        ds_val = ds_val.map(_parse_example)
        ds_test = ds_test.map(_parse_example)

        # prepare the training validation and test datasets
        ds_train = ds_train.shuffle(buffer_size)
        ds_train = ds_train.batch(batch_size=batch_size)
        ds_train = ds_train.repeat(-1)
        ds_val = ds_val.batch(batch_size=batch_size)
        num_test = sum(1 for _ in ds_test)
        ds_test = ds_test.batch(batch_size=num_test)

        return ds_train, ds_val, ds_test, ds_info
