import logging
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from visualization.activity_recogonition import *
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import *

"""Determines the basic information of the model

Parameters:
    model_name (string): name of the model (name list: 'Sequence_LSTM', 'Sequence_BiLSTM', 'Sequence_GRU', 'Sequence_BiGRU', 'Sequence_Conv1D',
                         'Sequence_BiConv1D', 'Sequence_Ensemble', 'Seq2Seq', 'Sequence_RNN_Fourier') 
    windows_size (int): length of sequence window (Default is 250)
    num_categories (int): number of label category (must be 12 or 13, when use 12, then remove the data with label 0)
    train (bool): if you want to train the model, set to True; if you want to evaluate (and visualize) the model, set to False
    folder (string): the folder, which contains the checkpoints, logs, summary, configs and deep visualization images of the model (Default is 'Sequence Prediction RNN')
"""

model_name = 'Sequence_BiConv1D'
windows_size = 250
num_categories = 12
train = True
folder = 'Sequence Prediction RNN'

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', train, 'Specify whether to train or evaluate a model.')


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder(folder)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load(num_categories=num_categories)

    # setup model
    if model_name == 'Sequence_LSTM':
        model = sequence_LSTM_model(input_shape=[windows_size, 6], num_categories=num_categories)
    elif model_name == 'Sequence_BiLSTM':
        model = sequence_BiLSTM_model(input_shape=[windows_size, 6], num_categories=num_categories)
    elif model_name == 'Sequence_GRU':
        model = sequence_GRU_model(input_shape=[windows_size, 6], num_categories=num_categories)
    elif model_name == 'Sequence_BiGRU':
        model = sequence_BiGRU_model(input_shape=[windows_size, 6], num_categories=num_categories)
    elif model_name == 'Sequence_Conv1D':
        model = sequence_Conv1D_model(input_shape=[windows_size, 6], num_categories=num_categories)
    elif model_name == 'Sequence_BiConv1D':
        model = sequence_BiConv1D_model(input_shape=[windows_size, 6], num_categories=num_categories)
    elif model_name == 'Sequence_Ensemble':
        model = sequence_Ensemble_model(input_shape=[windows_size, 6], num_categories=num_categories)
    elif model_name == 'Seq2Seq':
        model = Seq2Seq(num_categories=num_categories)
    elif model_name == 'Sequence_RNN_Fourier':
        model = sequence_RNN_Fourier_model(input_shape=[windows_size, 6], num_categories=num_categories)
    else:
        model = sequence_LSTM_model(input_shape=[windows_size, 6], num_categories=num_categories)

    if FLAGS.train:
        # set training loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # train the model
        trainer = Trainer(model, ds_train, ds_val, ds_info, model_name, run_paths=run_paths)
        for _ in trainer.train():
            continue
    else:
        # set validation loggers
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)

        # evaluate the model
        evaluate(model, ds_test, ds_info, model_name, run_paths=run_paths, num_categories=num_categories)
        visulization(model, run_paths, ds_test, model_name, num_categories=num_categories)


if __name__ == "__main__":
    app.run(main)
