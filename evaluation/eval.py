import logging
import gin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn import metrics
from evaluation.metrics import *
from scipy.ndimage import percentile_filter


@gin.configurable
def evaluate(model, ds_test, ds_info, model_name, run_paths, num_categories=13):
    """evaluate performance of the model

    Parameters:
        model (keras.Model): keras model object to be evaluated
        ds_test (tf.data.Dataset): test set
        ds_info (dictionary): information and structure of dataset
        model_name (string): name of the model (name list: 'Sequence_LSTM', 'Sequence_BiLSTM', 'Sequence_GRU', 'Sequence_BiGRU', 'Sequence_Conv1D',
                         'Sequence_BiConv1D', 'Sequence_Ensemble', 'Seq2Seq', 'Sequence_RNN_Fourier')
        run_paths (dictionary): storage path of model information
        num_categories (int): number of label category (must be 12 or 13, when use 12, then remove the data with label 0)
    """

    # set up the model and load the checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(), net=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, run_paths["path_ckpts_train"], max_to_keep=10)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    step = int(checkpoint.step.numpy())

    if model_name == 'Seq2Seq':
        encoder = model[0]
        decoder = model[1]
        test_loss = []
        test_accuracy = []
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # evaluate the model
        for idx, (test_windows, test_labels) in enumerate(ds_test):
            loss = 0
            enc_output, enc_hidden = encoder(test_windows)
            dec_hidden = enc_hidden
            dec_input = tf.zeros([test_labels.shape[0], 1], dtype=tf.int64)
            for t in range(test_labels.shape[1]):
                prediction, dec_state, _ = decoder(dec_input, dec_hidden, enc_output)
                loss += loss_object(tf.expand_dims(test_labels[:, t], 1), prediction)
                dec_input = tf.expand_dims(tf.argmax(prediction, axis=1), 1)
                if t == 0:
                    predictions = tf.expand_dims(prediction, 1)
                else:
                    predictions = tf.concat([predictions, tf.expand_dims(prediction, 1)], axis=1)
            test_loss.append(tf.math.reduce_mean(loss / int(test_labels.shape[1])).numpy())
            test_accuracy.append(metrics.accuracy_score(tf.reshape(test_labels, [-1]).numpy(), tf.reshape(tf.argmax(predictions, axis=2), [-1]).numpy()))
            if idx == 0:
                test_confusion_matrix = metrics.confusion_matrix(tf.reshape(test_labels, [-1]).numpy(), tf.reshape(tf.argmax(predictions, axis=2), [-1]).numpy())
        test_loss = np.mean(test_loss)
        test_accuracy = np.mean(test_accuracy)

        # log the evaluation information
        logging.info(f"Evaluating at step: {step}...")
        logging.info('loss:\n{}'.format(test_loss))
        logging.info('accuracy:\n{}'.format(test_accuracy))
        logging.info('confusion_matrix:\n{}'.format(test_confusion_matrix))
    else:
        # compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[[Accuracy()], [ConfusionMatrix(num_categories=num_categories)]])

        # evaluate the model
        result = model.evaluate(ds_test, return_dict=True)

        # log the evaluation information
        logging.info(f"Evaluating at step: {step}...")
        for key, value in result.items():
            logging.info('{}:\n{}'.format(key, value))

        for idx, (test_windows, test_labels) in enumerate(ds_test):
            predictions = model(test_windows)
            predictions = tf.argmax(predictions, axis=2)
            predictions = np.concatenate(predictions.numpy()).flatten()
            test_labels = np.concatenate(test_labels.numpy()).flatten()

            # postprocess the predictions by using the median filter or percentile filter
            # (also compare the results of filter and choose the best)
            plt.figure(dpi=800)
            plt.title('POSTPROCESSING METHODS COMPARISON')
            plt.xlabel('FILTER SIZE')
            plt.ylabel('ACCURACY(%)')
            plt.grid(b=True, axis='y')
            for percentile in range(45, 60, 5):
                size_list = range(0, 255, 10)
                acc_list = []
                for size in size_list:
                    if size != 0:
                        test_predictions = percentile_filter(predictions, percentile=percentile, size=size)
                    else:
                        test_predictions = predictions
                    test_accuracy = metrics.accuracy_score(test_labels, test_predictions) * 100
                    logging.info('accuracy(percentile filter {} with size {}):\n{}'.format(percentile, size, test_accuracy))
                    acc_list.append(test_accuracy)
                if percentile == 50:
                    plt.plot(size_list, acc_list, marker="s", markersize=3, label=(str(percentile) + '%' + ' Percentile Filter(Median Filter)'))
                else:
                    plt.plot(size_list, acc_list, marker="s", markersize=3, label=(str(percentile) + '%' + ' Percentile Filter'))
            plt.legend(loc='lower left')
            plt.savefig(run_paths['path_model_id'] + '/logs/eval/postprocessing_plot.png', dpi=800)
            plt.show()

            # plot the confusion matrix
            cm = result['confusion_matrix']
            fig, ax = plt.subplots()
            im = ax.imshow(cm + 1, norm=colors.LogNorm(vmin=100, vmax=cm.max()), cmap='Wistia')
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('NUMBER OF SAMPLING POINTS', rotation=-90, va="bottom")
            ax.set_xticks(np.arange(num_categories))
            ax.set_yticks(np.arange(num_categories))
            ax.set_xticklabels(['W', 'WU', 'WD', 'SI', 'ST', 'L', 'ST2SI', 'SI2ST', 'SI2L', 'L2SI', 'ST2L', 'L2ST'])
            ax.set_yticklabels(['W', 'WU', 'WD', 'SI', 'ST', 'L', 'ST2SI', 'SI2ST', 'SI2L', 'L2SI', 'ST2L', 'L2ST'])
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            for i in range(num_categories):
                for j in range(num_categories):
                    text = ax.text(j, i, cm[i, j], fontsize='x-small', ha="center", va="center", color="b")
            ax.set_title("SEQUENCE TO SEQUENCE CONFUSION MATRIX")
            fig.tight_layout()
            plt.show()
