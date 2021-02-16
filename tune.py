import logging
import ray
from ray import tune

from train import Trainer
from evaluation.metrics import *
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
    config_path (string): path of config.gin (must be an absolute path on the server)
"""

model_name = 'Sequence_BiLSTM'
windows_size = 250
num_categories = 12
config_path = '/home/RUS_CIP/st169530/dl-lab-2020-team09/human_activity_sequence_prediction(accomplished by Peizheng Li)/configs/tuning_config.gin'


def tuning(config):
    # hyperparameters
    bindings = []
    for key, value in config.items():
        bindings.append('{}={}'.format(str(key), str(value)))

    # generate folder structures
    run_paths = utils_params.gen_run_folder(','.join(bindings))

    # gin-config
    gin.parse_config_files_and_bindings([config_path], bindings)
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

    # set training loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # train the model
    trainer = Trainer(model, ds_train, ds_val, ds_info, model_name, run_paths=run_paths)
    for val_accuracy in trainer.train():
        tune.report(val_accuracy=val_accuracy * 100)

    # set validation loggers
    utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)

    # evaluate the model
    trained_model = trainer.model_output()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[[Accuracy()], [ConfusionMatrix(num_categories=num_categories)]])

    result = trained_model.evaluate(ds_test, return_dict=True)
    test_accuracy = result['accuracy']
    visulization(model, run_paths, ds_test, model_name, num_categories=num_categories)
    tune.report(test_accuracy=test_accuracy * 100)


# initialize ray
ray.init()

# run the training program
analysis = tune.run(tuning,
                    name="tuning",
                    local_dir="./ray_results",
                    num_samples=1,
                    resources_per_trial={"cpu": 48, "gpu": 1},
                    config={"folder": tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                            "sequence_BiLSTM_model.dropout_rate": tune.quniform(0.3, 0.5, 0.05)})

# print the best result
print("Best config is:", analysis.get_best_config(metric="test_accuracy", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()
