import tensorflow as tf


class Accuracy(tf.keras.metrics.Metric):
    """metric: accuracy"""
    def __init__(self, name='accuracy', **kwargs):
        super(Accuracy, self).__init__(name=name, **kwargs)
        # initialize parameters
        self.accuracy = self.add_weight(name='accuracy', initializer='zeros')

    def update_state(self, labels, predictions, *args, **kwargs):
        # update parameters
        self.predictions = tf.reshape(tf.argmax(predictions, axis=2), [-1])
        self.labels = tf.reshape(labels, [-1])

        self.accuracy = tf.math.reduce_mean(tf.cast(tf.math.equal(self.labels, self.predictions), dtype=tf.float32))

    def result(self):
        # return parameters
        return self.accuracy


class ConfusionMatrix(tf.keras.metrics.Metric):
    """metric: confusion matrix"""
    def __init__(self, num_categories=13, name='confusion_matrix', **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        # initialize parameters
        self.num_categories = num_categories
        self.confusion_matrix = self.add_weight(name='confusion_matrix', shape=(self.num_categories, self.num_categories), initializer='zeros')

    def update_state(self, labels, predictions, *args, **kwargs):
        # update parameters
        self.predictions = tf.reshape(tf.argmax(predictions, axis=2), [-1])
        self.labels = tf.reshape(labels, [-1])

        self.confusion_matrix = tf.math.confusion_matrix(tf.squeeze(self.labels), tf.squeeze(self.predictions), num_classes=self.num_categories, dtype=tf.int32)

    def result(self):
        # return parameters
        return self.confusion_matrix
