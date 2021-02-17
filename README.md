# Human-Activity-Recognition
Human Activity Recognition (HAR) is a problem that is an active research field in pervasive computing. For this project, HAPT dataset is chosen, which is recorded through the inertial sensors (accelerometer and gyroscope). We have developed various deep residual neural networks as well as some convolutional neural networks. On this basis, the activity signal is recognized by ensemble learning through time- and frequency-domain combination. Our model has finally achieved an accuracy of 96.5%.

## Content
- Input pipeline with TFRecord of dataset HATP;
- Two kinds of model types(Sequence2Label and Sequence2Sequence) and each of them contain various modes(LSTM, GRU, Con1D, Encoder-Decoder);
- Different kinds of metrics (accuracy, confusion-matrix, precision and recall) and corresponding evaluations;
- Feature enhancement (Fast Fourier Transform);
- Signal preprocessing (normalization) and balancing (Oversampling and Undersampling);
- Visualization

## How to run the code
-  Download the original data file and modify the parameter *load.data_dir* in `config.gin` and `tuning_config.gin` to your corresponding data directory
   - Human Activities and Postural Transitions(HAPT) Dataset: https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
- If you want to start training a new model, run `main.py` directly (To evaluate the model, just change the parameter train in main.py to False)
- If you want to start fine-tuning of the model, run `tune.py` directly

Note: If you want to change the type and parameters of the model, you need to modify the parameters in `config.gin`, `tuning_config.gin`, `main.py` and `tune.py`

Note: The default parameters are: BiConv1D model for `main.py`; BiLSTM model for `tune.py`

## Results
Model Name|Accuracy
----------|--------
LSTM|93.1%
BiLSTM|93.0%
GRU|90.8%
BiGRU|93.9%
Conv1D|93.9%
**BiConv1D**|**96.5%**
Ensemble|92.8%
Ensemble_Fourier|91.4%

Bidirectional Conv1D Encoder-Decoder Model shows the best performance.
