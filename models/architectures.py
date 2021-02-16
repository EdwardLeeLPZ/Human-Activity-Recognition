from models.layers import *


@gin.configurable
def sequence_LSTM_model(input_shape, dropout_rate, num_categories=13):
    """Defines a LSTM architecture with sequence output.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        dropout_rate (float): dropout rate of dropout layers in the output block
        num_categories (int): number of label category (must be 12 or 13, when use 12, then remove the data with label 0)

    Returns:
        (keras.Model): keras model object
    """

    # set the input
    inputs = tf.keras.Input(input_shape)

    # establish feature extraction blocks
    outputs = basic_LSTM(inputs, 256)
    outputs = basic_Dense(outputs, 128)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = basic_LSTM(outputs, 128)
    outputs = basic_Dense(outputs, 64)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = basic_LSTM(outputs, 64)
    outputs = basic_Dense(outputs, 32)

    # establish output layers
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = tf.keras.layers.Dense(num_categories, activation='softmax')(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='sequence_LSTM_model')


@gin.configurable
def sequence_GRU_model(input_shape, dropout_rate, num_categories=13):
    """Defines a GRU architecture with sequence output.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        dropout_rate (float): dropout rate of dropout layers in the output block
        num_categories (int): number of label category (must be 12 or 13, when use 12, then remove the data with label 0)

    Returns:
        (keras.Model): keras model object
    """

    # set the input
    inputs = tf.keras.Input(input_shape)

    # establish feature extraction blocks
    outputs = basic_GRU(inputs, 256)
    outputs = basic_Dense(outputs, 128)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = basic_GRU(outputs, 128)
    outputs = basic_Dense(outputs, 64)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = basic_GRU(outputs, 64)
    outputs = basic_Dense(outputs, 32)

    # establish output layers
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = tf.keras.layers.Dense(num_categories, activation='softmax')(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='sequence_GRU_model')


@gin.configurable
def sequence_Conv1D_model(input_shape, dropout_rate, initial_units=64, block_number=3, padding_number=3, num_categories=13):
    """Defines a 1D convolutional architecture with sequence output.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        dropout_rate (float): dropout rate of dropout layers in the output block
        initial_units (int): number of base filters, which are doubled for every VGG block (Default is 64)
        block_number (int): number of convolutional blocks (Default is 3)
        padding_number (int): number of padding (Default is 3)
        num_categories (int): number of label category (must be 12 or 13, when use 12, then remove the data with label 0)

    Returns:
        (keras.Model): keras model object
    """

    # set the input and perform padding to change the length of sequence into 256 (easy to compress)
    inputs = tf.keras.Input(input_shape)
    outputs = tf.keras.layers.ZeroPadding1D(padding=padding_number)(inputs)

    # establish the encoder
    for i in range(block_number):
        outputs = basic_Conv1D(outputs, initial_units * (2 ** i), 3, strides=1, padding='same')
        outputs = basic_Conv1D(outputs, initial_units * (2 ** i), 3, strides=1, padding='same')
        outputs = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(outputs)
        outputs = tf.keras.layers.Dropout(0.2)(outputs)
    # establish the layer of feature map
    outputs = basic_Conv1D(outputs, initial_units * (2 ** block_number), 3, strides=1, padding='same')
    outputs = basic_Conv1D(outputs, initial_units * (2 ** block_number), 3, strides=1, padding='same')
    # establish the decoder
    for i in range(block_number):
        outputs = tf.keras.layers.UpSampling1D(size=2)(outputs)
        outputs = basic_Conv1D(outputs, initial_units * (2 ** (block_number - 1 - i)), 3, strides=1, padding='same')
        outputs = basic_Conv1D(outputs, initial_units * (2 ** (block_number - 1 - i)), 3, strides=1, padding='same')
        outputs = tf.keras.layers.Dropout(0.2)(outputs)
    outputs = outputs[:, padding_number:-padding_number, :]

    # establish output layers
    outputs = basic_Dense(outputs, 32)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = tf.keras.layers.Dense(num_categories, activation='softmax')(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='sequence_Conv1D_model')


@gin.configurable
def sequence_BiLSTM_model(input_shape, dropout_rate, num_categories=13):
    """Defines a bidirectional LSTM architecture with sequence output.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        dropout_rate (float): dropout rate of dropout layers in the output block
        num_categories (int): number of label category (must be 12 or 13, when use 12, then remove the data with label 0)

    Returns:
        (keras.Model): keras model object
    """

    # set the input
    inputs = tf.keras.Input(input_shape)

    # establish feature extraction blocks
    outputs = basic_BiLSTM(inputs, 256)
    outputs = basic_Dense(outputs, 128)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = basic_BiLSTM(outputs, 128)
    outputs = basic_Dense(outputs, 64)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = basic_BiLSTM(outputs, 64)
    outputs = basic_Dense(outputs, 32)

    # establish output layers
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = tf.keras.layers.Dense(num_categories, activation='softmax')(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='sequence_BiLSTM_model')


@gin.configurable
def sequence_BiGRU_model(input_shape, dropout_rate, num_categories=13):
    """Defines a bidirectional GRU architecture with sequence output.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        dropout_rate (float): dropout rate of dropout layers in the output block
        num_categories (int): number of label category (must be 12 or 13, when use 12, then remove the data with label 0)

    Returns:
        (keras.Model): keras model object
    """

    # set the input
    inputs = tf.keras.Input(input_shape)

    # establish feature extraction blocks
    outputs = basic_BiGRU(inputs, 256)
    outputs = basic_Dense(outputs, 128)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = basic_BiGRU(outputs, 128)
    outputs = basic_Dense(outputs, 64)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = basic_BiGRU(outputs, 64)
    outputs = basic_Dense(outputs, 32)

    # establish output layers
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = tf.keras.layers.Dense(num_categories, activation='softmax')(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='sequence_BiGRU_model')


@gin.configurable
def sequence_BiConv1D_model(input_shape, dropout_rate, initial_units=64, block_number=3, padding_number=3, num_categories=13):
    """Defines a bidirectional 1D convolutional architecture with sequence output.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        dropout_rate (float): dropout rate of dropout layers in the output block
        initial_units (int): number of base filters, which are doubled for every VGG block (Default is 64)
        block_number (int): number of convolutional blocks (Default is 3)
        padding_number (int): number of padding (Default is 3)
        num_categories (int): number of label category (must be 12 or 13, when use 12, then remove the data with label 0)

    Returns:
        (keras.Model): keras model object
    """

    # set the input and perform padding to change the length of sequence into 512 (easy to compress)
    inputs = tf.keras.Input(input_shape)
    reversed_inputs = inputs[:, ::-1, :]
    bidirectional_inputs = tf.concat([inputs, reversed_inputs], axis=1)
    outputs = tf.keras.layers.ZeroPadding1D(padding=(padding_number * 2))(bidirectional_inputs)

    # establish the encoder
    for i in range(block_number):
        outputs = basic_Conv1D(outputs, initial_units * (2 ** i), 3, strides=1, padding='same')
        outputs = basic_Conv1D(outputs, initial_units * (2 ** i), 3, strides=1, padding='same')
        outputs = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(outputs)
        outputs = tf.keras.layers.Dropout(0.2)(outputs)
    # establish the layer of feature map
    outputs = basic_Conv1D(outputs, initial_units * (2 ** block_number), 3, strides=1, padding='same')
    outputs = basic_Conv1D(outputs, initial_units * (2 ** block_number), 3, strides=1, padding='same')
    # establish the decoder
    for i in range(block_number):
        outputs = tf.keras.layers.UpSampling1D(size=2)(outputs)
        outputs = basic_Conv1D(outputs, initial_units * (2 ** (block_number - 1 - i)), 3, strides=1, padding='same')
        outputs = basic_Conv1D(outputs, initial_units * (2 ** (block_number - 1 - i)), 3, strides=1, padding='same')
        outputs = tf.keras.layers.Dropout(0.2)(outputs)
    o1, o2 = tf.split(outputs, num_or_size_splits=2, axis=1)
    outputs = o1 + o2
    outputs = outputs[:, padding_number:-padding_number, :]
    outputs = basic_Conv1D(outputs, initial_units, 3, strides=1, padding='same')
    outputs = basic_Conv1D(outputs, initial_units, 3, strides=1, padding='same')

    # establish output layers
    outputs = basic_Dense(outputs, 32)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = tf.keras.layers.Dense(num_categories, activation='softmax')(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='sequence_BiConv1D_model')


@gin.configurable
def sequence_Ensemble_model(input_shape, dropout_rate, num_categories=13):
    """Defines a ensemble learning architecture with sequence output.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        dropout_rate (float): dropout rate of dropout layers in the output block
        num_categories (int): number of label category (must be 12 or 13, when use 12, then remove the data with label 0)

    Returns:
        (keras.Model): keras model object
    """

    # set the input
    inputs = tf.keras.Input(input_shape)

    # establish the bidirectional LSTM model, bidirectional GRU model and bidirectional 1D convolutional model
    BiLSTM_model = sequence_BiLSTM_model(input_shape=input_shape, num_categories=num_categories)
    BiGRU_model = sequence_BiGRU_model(input_shape=input_shape, num_categories=num_categories)
    BiConv1D_model = sequence_BiConv1D_model(input_shape=input_shape, num_categories=num_categories)
    outputs_BiLSTM = BiLSTM_model(inputs)
    outputs_BiGRU = BiGRU_model(inputs)
    outputs_BiConv1D = BiConv1D_model(inputs)

    # fuse the outputs of different models together
    outputs = tf.concat([outputs_BiLSTM, outputs_BiGRU, outputs_BiConv1D], axis=2)

    # establish output layers
    outputs = basic_Dense(outputs, num_categories * 3)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = basic_Dense(outputs, num_categories * 2)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = tf.keras.layers.Dense(num_categories, activation='softmax')(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='sequence_Ensemble_model')


@gin.configurable
def sequence_RNN_Fourier_model(input_shape, dropout_rate, num_categories=13):
    """Defines a ensemble learning through time- and frequency-domain combination with sequence output.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        dropout_rate (float): dropout rate of dropout layers in the output block
        num_categories (int): number of label category (must be 12 or 13, when use 12, then remove the data with label 0)

    Returns:
        (keras.Model): keras model object
    """

    # set the input
    inputs = tf.keras.Input(input_shape)

    # establish the bidirectional LSTM model with time domain sequence
    BiLSTM_model = sequence_BiLSTM_model(input_shape=input_shape, num_categories=num_categories)
    outputs = BiLSTM_model(inputs)

    # establish the bidirectional 1D convolutional model with frequency domain signal
    inputs_fourier = tf.transpose(tf.cast(tf.transpose(inputs, perm=[0, 2, 1]), dtype=tf.complex64), perm=[0, 2, 1])
    Conv1D_model = sequence_Conv1D_model(input_shape=input_shape, num_categories=num_categories)
    outputs_fourier = Conv1D_model(inputs_fourier)

    # fuse the outputs of different models together
    outputs = tf.concat([outputs, outputs_fourier], axis=2)

    # establish output layers
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = basic_Dense(outputs, num_categories * 2)
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = tf.keras.layers.Dense(num_categories, activation='softmax')(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='sequence_RNN_Fourier_model')


@gin.configurable
def Seq2Seq(units, num_categories=13):
    """Defines a seq2seq model with sequence output.

    Parameters:
        units (int): number of base filters
        num_categories (int): number of label category (must be 12 or 13, when use 12, then remove the data with label 0)

    Returns:
        (keras.Model): keras model object
    """

    # establish the encoder
    encoder = EncoderCell(units)

    # establish the decoder
    decoder = DecoderCell(units, num_categories=num_categories)

    return [encoder, decoder]
