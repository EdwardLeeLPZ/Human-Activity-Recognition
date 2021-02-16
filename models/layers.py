import gin
import tensorflow as tf


@gin.configurable
def basic_Dense(inputs, units, use_bn=True, use_activation=True):
    """A single dense layer

    Parameters:
        inputs (Tensor): input of the dense layer
        units (int): number of filters used for the dense layer
        use_bn (bool): whether the batch normalization is used or not (Default is True)
        use_activation (bool): whether the non-linear activation is used or not (Default is True)

    Returns:
        (Tensor): output of the single dense layer
    """

    outputs = tf.keras.layers.Dense(units, activation='linear')(inputs)
    if use_bn:
        outputs = tf.keras.layers.BatchNormalization()(outputs)
    if use_activation:
        outputs = tf.keras.layers.Activation('tanh')(outputs)
    return outputs


@gin.configurable
def basic_LSTM(inputs, units, return_sequences=True, use_bn=True, use_activation=True):
    """A single LSTM layer

    Parameters:
        inputs (Tensor): input of the LSTM layer
        units (int): number of filters used for the LSTM layer
        return_sequences (bool): whether output is sequence or not (Default is True)
        use_bn (bool): whether the batch normalization is used or not (Default is True)
        use_activation (bool): whether the non-linear activation is used or not (Default is True)

    Returns:
        (Tensor): output of the single LSTM layer
    """

    outputs = tf.keras.layers.LSTM(units, return_sequences=return_sequences)(inputs)
    if use_bn:
        outputs = tf.keras.layers.BatchNormalization()(outputs)
    if use_activation:
        outputs = tf.keras.layers.Activation('tanh')(outputs)
    return outputs


@gin.configurable
def basic_GRU(inputs, units, return_sequences=True, use_bn=True, use_activation=True):
    """A single GRU layer

    Parameters:
        inputs (Tensor): input of the GRU layer
        units (int): number of filters used for the GRU layer
        return_sequences (bool): whether output is sequence or not (Default is True)
        use_bn (bool): whether the batch normalization is used or not (Default is True)
        use_activation (bool): whether the non-linear activation is used or not (Default is True)

    Returns:
        (Tensor): output of the single GRU layer
    """

    outputs = tf.keras.layers.GRU(units, return_sequences=return_sequences)(inputs)
    if use_bn:
        outputs = tf.keras.layers.BatchNormalization()(outputs)
    if use_activation:
        outputs = tf.keras.layers.Activation('tanh')(outputs)
    return outputs


@gin.configurable
def basic_Conv1D(inputs, filters, kernel_size, strides=1, padding='same', use_bn=True, use_activation=True):
    """A single convolutional layer

    Parameters:
        inputs (Tensor): input of the convolutional layer
        filters (int): number of filters used for the convolutional layer
        kernel_size (tuple: 2): kernel size used for the convolutional layer, e.g. (3, 3)
        strides (int): stride of the convolutional layer (Default is 1)
        padding (string): padding type of the convolutional layer (Default is 'same')
        use_bn (bool): whether the batch normalization is used or not (Default is True)
        use_activation (bool): whether the non-linear activation is used or not (Default is True)

    Returns:
        (Tensor): output of the single convolutional layer
    """

    outputs = tf.keras.layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, activation='linear')(inputs)
    if use_bn:
        outputs = tf.keras.layers.BatchNormalization()(outputs)
    if use_activation:
        outputs = tf.keras.layers.Activation('tanh')(outputs)
    return outputs


@gin.configurable
def basic_BiLSTM(inputs, units, return_sequences=True, use_bn=True, use_activation=True):
    """A single bidirectional LSTM layer

    Parameters:
        inputs (Tensor): input of the bidirectional LSTM layer
        units (int): number of filters used for the bidirectional LSTM layer
        return_sequences (bool): whether output is sequence or not (Default is True)
        use_bn (bool): whether the batch normalization is used or not (Default is True)
        use_activation (bool): whether the non-linear activation is used or not (Default is True)

    Returns:
        (Tensor): output of the single bidirectional LSTM layer
    """

    outputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=return_sequences), merge_mode='concat')(inputs)
    if use_bn:
        outputs = tf.keras.layers.BatchNormalization()(outputs)
    if use_activation:
        outputs = tf.keras.layers.Activation('tanh')(outputs)
    return outputs


@gin.configurable
def basic_BiGRU(inputs, units, return_sequences=True, use_bn=True, use_activation=True):
    """A single bidirectional GRU layer

    Parameters:
        inputs (Tensor): input of the bidirectional GRU layer
        units (int): number of filters used for the bidirectional GRU layer
        return_sequences (bool): whether output is sequence or not (Default is True)
        use_bn (bool): whether the batch normalization is used or not (Default is True)
        use_activation (bool): whether the non-linear activation is used or not (Default is True)

    Returns:
        (Tensor): output of the single bidirectional GRU layer
    """

    outputs = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units, return_sequences=return_sequences), merge_mode='concat')(inputs)
    if use_bn:
        outputs = tf.keras.layers.BatchNormalization()(outputs)
    if use_activation:
        outputs = tf.keras.layers.Activation('tanh')(outputs)
    return outputs


class EncoderCell(tf.keras.Model):
    """An encoder cell"""

    def __init__(self, enc_units):
        super(EncoderCell, self).__init__()
        self.enc_units = enc_units
        # use 2 convolutional layers to extract feature
        self.conv1 = tf.keras.layers.Conv1D(enc_units / 4, 5, strides=1, padding='same', activation='linear')
        self.bn_cv1 = tf.keras.layers.BatchNormalization()
        self.tanh_cv1 = tf.keras.layers.Activation('tanh')
        self.conv2 = tf.keras.layers.Conv1D(enc_units / 2, 5, strides=1, padding='same', activation='linear')
        self.bn_cv2 = tf.keras.layers.BatchNormalization()
        self.tanh_cv2 = tf.keras.layers.Activation('tanh')
        # use 1 GRU layers to build the feature map and feature tensor
        self.gru_out = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    @tf.function
    def call(self, x):
        # use 2 convolutional layers to extract feature
        x = self.conv1(x)
        x = self.bn_cv1(x)
        x = self.tanh_cv1(x)
        x = self.conv2(x)
        x = self.bn_cv2(x)
        x = self.tanh_cv2(x)
        # use 1 GRU layers to build the feature map and feature tensor
        output, state = self.gru_out(x)
        return output, state


class BahdanauAttention(tf.keras.layers.Layer):
    """An Bahdanau attention cell"""

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class DecoderCell(tf.keras.Model):
    """An encoder cell"""

    def __init__(self, dec_units, num_categories=13):
        super(DecoderCell, self).__init__()
        self.dec_units = dec_units
        self.num_categories = num_categories
        # calculate the attention from the feature map
        self.attention = BahdanauAttention(self.dec_units)
        self.embedding = tf.keras.layers.Embedding(self.num_categories, self.num_categories)
        # use 2 GRU layers to rebuild the prediction
        self.gru_in = tf.keras.layers.GRU(self.dec_units, return_sequences=True)
        self.bn_in = tf.keras.layers.BatchNormalization()
        self.dense = tf.keras.layers.Dense(self.dec_units, activation='linear')
        self.bn_out = tf.keras.layers.BatchNormalization()
        self.tanh = tf.keras.layers.Activation('tanh')
        self.gru_out = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

        self.dense_out = tf.keras.layers.Dense(self.num_categories, activation='softmax')

    @tf.function
    def call(self, x, hidden, enc_output):
        # calculate the attention from the feature map
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        # fuse the input and the attention together
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # use 2 GRU layers to rebuild the prediction
        x = self.gru_in(x)
        x = self.bn_in(x)
        x = self.dense(x)
        x = self.bn_out(x)
        x = self.tanh(x)
        output, state = self.gru_out(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.dense_out(output)
        return x, state, attention_weights
