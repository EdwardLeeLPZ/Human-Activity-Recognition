import gin
import logging
from evaluation.metrics import *


@gin.configurable
class Trainer(object):
    """Trainer for the model training"""

    def __init__(self, model, ds_train, ds_val, ds_info, model_name, run_paths, total_steps, log_interval, ckpt_interval, learning_rate=1e-3):
        self.model = model
        self.model_name = model_name
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval  # step size for logging
        self.ckpt_interval = ckpt_interval  # step size for saving checkpoints

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # loss objective and metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.train_accuracy = tf.keras.metrics.Accuracy(name='train_sequence_accuracy')
        self.val_accuracy = tf.keras.metrics.Accuracy(name='val_sequence_accuracy')
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # summary writer
        self.train_summary_writer = tf.summary.create_file_writer(self.run_paths['path_summary_train'])
        self.val_summary_writer = tf.summary.create_file_writer(self.run_paths['path_summary_val'])

        # checkpoint manager
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.run_paths["path_ckpts_train"], max_to_keep=10)

    @tf.function
    def train_step(self, windows, labels):
        """one-step training"""

        with tf.GradientTape() as tape:
            predictions = self.model(windows, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        # calculate the training loss and training metrics
        self.train_loss(loss)
        self.train_accuracy(tf.reshape(labels, [-1]), tf.reshape(tf.argmax(predictions, axis=2), [-1]))

    def train_step_Seq2Seq(self, windows, labels):
        """one-step training of Seq2Seq model"""

        encoder = self.model[0]
        decoder = self.model[1]
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(windows)
            dec_hidden = enc_hidden
            dec_input = tf.zeros([labels.shape[0], 1], dtype=tf.int64)
            # each time predict 1 label of 1 time point, 250 step in total
            for t in range(labels.shape[1]):
                prediction, dec_state, _ = decoder(dec_input, dec_hidden, enc_output)
                loss += self.loss_object(tf.expand_dims(labels[:, t], 1), prediction)
                dec_input = tf.expand_dims(labels[:, t], 1)
                if t == 0:
                    predictions = tf.expand_dims(prediction, 1)
                else:
                    predictions = tf.concat([predictions, tf.expand_dims(prediction, 1)], axis=1)
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        # calculate the training loss and training metrics
        self.train_loss(loss / int(labels.shape[1]))
        self.train_accuracy(tf.reshape(labels, [-1]), tf.reshape(tf.argmax(predictions, axis=2), [-1]))

    @tf.function
    def val_step(self, windows, labels):
        """one-step validation"""

        predictions = self.model(windows, training=False)
        v_loss = self.loss_object(labels, predictions)
        # calculate the validation loss and validation metrics
        self.val_loss(v_loss)
        self.val_accuracy(tf.reshape(labels, [-1]), tf.reshape(tf.argmax(predictions, axis=2), [-1]))

    def val_step_Seq2Seq(self, windows, labels):
        """one-step validation of Seq2Seq model"""

        encoder = self.model[0]
        decoder = self.model[1]
        loss = 0
        enc_output, enc_hidden = encoder(windows)
        dec_hidden = enc_hidden
        dec_input = tf.zeros([labels.shape[0], 1], dtype=tf.int64)
        # each time predict 1 label of 1 time point, 250 step in total
        for t in range(labels.shape[1]):
            prediction, dec_state, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += self.loss_object(tf.expand_dims(labels[:, t], 1), prediction)
            dec_input = tf.expand_dims(tf.argmax(prediction, axis=1), 1)
            if t == 0:
                predictions = tf.expand_dims(prediction, 1)
            else:
                predictions = tf.concat([predictions, tf.expand_dims(prediction, 1)], axis=1)
        # calculate the validation loss and validation metrics
        self.val_loss(loss / int(labels.shape[1]))
        self.val_accuracy(tf.reshape(labels, [-1]), tf.reshape(tf.argmax(predictions, axis=2), [-1]))

    def train(self):
        # record the current optimal accuracy and loss, which is used for early stopping
        max_accuracy_record = 0.0
        min_loss_record = float("inf")

        # if training is interrupted unexpectedly, resume the model from here and continue training
        # or if it is the first step of training, start training from the beginning
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
            self.ckpt.step.assign_add(1)
        else:
            print("Initializing from scratch.")

        for idx, (windows, labels) in enumerate(self.ds_train):
            step = int(self.ckpt.step.numpy())

            # perform one-step training
            if self.model_name == 'Seq2Seq':
                self.train_step_Seq2Seq(windows, labels)
            else:
                self.train_step(windows, labels)

            # Write train summary to tensorboard
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=step)
                tf.summary.scalar('accuracy', self.train_accuracy.result() * 100, step=step)

            # check if the model should be validated
            if int(step) % self.log_interval == 0:
                # reset validation metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                # perform one-step validation
                for val_windows, val_labels in self.ds_val:
                    if self.model_name == 'Seq2Seq':
                        self.val_step_Seq2Seq(val_windows, val_labels)
                    else:
                        self.val_step(val_windows, val_labels)

                # log the training and validation information
                template = 'Step {} [Training/Validation]: Loss: {:.5f}/{:.5f}, Accuracy: {:.5f}/{:.5f}'
                logging.info(template.format(step,
                                             self.train_loss.result(), self.val_loss.result(),
                                             self.train_accuracy.result() * 100, self.val_accuracy.result() * 100))
                # record the accuracy and loss of this step, which is used for early stopping
                accuracy_record = self.val_accuracy.result().numpy()
                loss_record = self.val_loss.result().numpy()

                # reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                # Write validation summary to tensorboard
                with self.val_summary_writer.as_default():
                    tf.summary.scalar('loss', self.val_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.val_accuracy.result() * 100, step=step)

                yield self.val_accuracy.result().numpy()

            # save checkpoints
            if int(self.ckpt.step) % self.ckpt_interval == 0:
                if accuracy_record > max_accuracy_record:
                    max_accuracy_record = accuracy_record
                    min_loss_record = loss_record
                    # Save checkpoint
                    save_path = self.ckpt_manager.save()
                    logging.info(f'Saved checkpoint for step {int(self.ckpt.step)} to {save_path}.')
                elif accuracy_record == max_accuracy_record and loss_record < min_loss_record:
                    max_accuracy_record = accuracy_record
                    min_loss_record = loss_record
                    # save checkpoint
                    save_path = self.ckpt_manager.save()
                    logging.info(f'Saved checkpoint for step {int(self.ckpt.step)} to {save_path}.')
                else:
                    logging.info(
                        f'Did not save checkpoint for step {int(self.ckpt.step)}, because the validation accuracy was not high enough.')

            # finish
            if int(step) % self.total_steps == 0:
                if accuracy_record > max_accuracy_record:
                    # Save final checkpoint
                    save_path = self.ckpt_manager.save()
                    logging.info(f'Finished training after {step} steps and saved final checkpoint to {save_path}.')
                elif accuracy_record == max_accuracy_record and loss_record < min_loss_record:
                    # save final checkpoint
                    save_path = self.ckpt_manager.save()
                    logging.info(f'Finished training after {step} steps and saved final checkpoint to {save_path}.')
                else:
                    logging.info(
                        f'Finished training after {step} steps, but did not save checkpoint for step {int(self.ckpt.step)}, because the validation accuracy was not high enough.')
                return self.val_accuracy.result().numpy()

            self.ckpt.step.assign_add(1)

    def model_output(self):
        """model output interface (used for fine tuning)"""

        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            tf.print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            tf.print("Initializing from scratch.")
        return self.model
