import matplotlib.pyplot as plt
import numpy as np
from models.architectures import *
from scipy.ndimage import median_filter


def visulization(model, run_paths, dataset, model_name, num_categories=13):
    """visualize  the model

    Parameters:
        model (keras.Model): keras model object to be evaluated
                run_paths (dictionary): storage path of model information
        dataset (tf.data.Dataset): dataset to be visualized
        model_name (string): name of the model (name list: 'Sequence_LSTM', 'Sequence_BiLSTM', 'Sequence_GRU', 'Sequence_BiGRU', 'Sequence_Conv1D',
                         'Sequence_BiConv1D', 'Sequence_Ensemble', 'Seq2Seq', 'Sequence_RNN_Fourier')
        num_categories (int): number of label category (must be 12 or 13, when use 12, then remove the data with label 0)
    """

    # set up the model and load the checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(), net=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, run_paths["path_ckpts_train"], max_to_keep=10)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)

    pred_list = []
    label_list = []
    acc_x, acc_y, acc_z = [], [], []
    gyro_x, gyro_y, gyro_z = [], [], []
    for idx, (window_sequence, label) in enumerate(dataset):
        # make the predictions
        if model_name == 'Seq2Seq':
            encoder = model[0]
            decoder = model[1]
            enc_output, enc_hidden = encoder(window_sequence)
            dec_hidden = enc_hidden
            dec_input = tf.zeros([label.shape[0], 1], dtype=tf.int64)
            for t in range(label.shape[1]):
                prediction, dec_state, _ = decoder(dec_input, dec_hidden, enc_output)
                dec_input = tf.expand_dims(tf.argmax(prediction, axis=1), 1)
                if t == 0:
                    predictions = tf.expand_dims(prediction, 1)
                else:
                    predictions = tf.concat([predictions, tf.expand_dims(prediction, 1)], axis=1)
            prediction = tf.argmax(predictions, axis=2)
        else:
            model.compile(optimizer=tf.keras.optimizers.Adam(),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy())
            prediction = model(window_sequence)
            prediction = tf.argmax(prediction, axis=2)
        prediction = np.concatenate(prediction.numpy())
        label = np.concatenate(label.numpy())
        pred_list.append(prediction)
        label_list.append(label)
        window_sequence = np.concatenate(window_sequence.numpy())
        acc_x.append(window_sequence[:, 0]), acc_y.append(window_sequence[:, 1]), acc_z.append(window_sequence[:, 2])
        gyro_x.append(window_sequence[:, 3]), gyro_y.append(window_sequence[:, 4]), gyro_z.append(window_sequence[:, 5])
        if idx >= 0:
            break
    acc_x, acc_y, acc_z = np.concatenate(acc_x), np.concatenate(acc_y), np.concatenate(acc_z)
    gyro_x, gyro_y, gyro_z = np.concatenate(gyro_x), np.concatenate(gyro_y), np.concatenate(gyro_z)

    # postprocess the predictions
    preds = np.concatenate(pred_list)
    preds = median_filter(preds, size=140)
    labels = np.concatenate(label_list)

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
    view = [0, 15000]
    acc_x, acc_y, acc_z = acc_x[view[0]:view[1]], acc_y[view[0]:view[1]], acc_z[view[0]:view[1]]
    gyro_x, gyro_y, gyro_z = gyro_x[view[0]:view[1]], gyro_y[view[0]:view[1]], gyro_z[view[0]:view[1]]
    preds = preds[view[0]:view[1]]
    labels = labels[view[0]:view[1]]

    # plot the data as well as  their predictions and labels (the first figure is generated from predicted value, the second is the true distribution)
    plt.figure(figsize=(12, 6), dpi=150)
    ax11 = plt.subplot(2, 1, 1)
    plt.title('PREDICTION', fontdict={'weight': 'normal', 'size': 'x-large'})
    plt.tick_params(labelsize='x-small')
    ax11.set_xlabel("TIME SEQUENCE")
    ax11.set_ylabel("NORMALIZED LINEAR ACCELERATION")
    ax12 = ax11.twinx()
    ax12.set_ylabel("NORMALIZED AUGULAR VELOCITY")
    plt.plot(acc_x, label='acc_x', linewidth=1)
    plt.plot(acc_y, label='acc_y', linewidth=1)
    plt.plot(acc_z, label='acc_z', linewidth=1)
    plt.plot(gyro_x, label='gyro_x', linewidth=1)
    plt.plot(gyro_y, label='gyro_y', linewidth=1)
    plt.plot(gyro_z, label='gyro_z', linewidth=1)
    labeling(preds, num_categories)
    ax21 = plt.subplot(2, 1, 2)
    plt.title('TRUE VALUES', fontdict={'weight': 'normal', 'size': 'x-large'})
    plt.tick_params(labelsize='x-small')
    ax21.set_xlabel("TIME SEQUENCE")
    ax11.set_ylabel("NORMALIZED LINEAR ACCELERATION")
    ax22 = ax21.twinx()
    ax22.set_ylabel("NORMALIZED AUGULAR VELOCITY")
    plt.plot(acc_x, label='acc_x', linewidth=1)
    plt.plot(acc_y, label='acc_y', linewidth=1)
    plt.plot(acc_z, label='acc_z', linewidth=1)
    plt.plot(gyro_x, label='gyro_x', linewidth=1)
    plt.plot(gyro_y, label='gyro_y', linewidth=1)
    plt.plot(gyro_z, label='gyro_z', linewidth=1)
    labeling(labels, num_categories)
    plt.tight_layout()
    plt.savefig(run_paths['path_model_id'] + '/plot.png', dpi=150)
    plt.show()
