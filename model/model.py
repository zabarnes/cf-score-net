import tensorflow as tf
import logging
import numpy as np
import os

from tqdm import trange

class Model(object):
    def __init__(self, dataset, network, params):
        logging.info("Creating the model...")

        self.dataset = dataset
        self.network = network
        self.params = params
        self.restore_from = None

        if not os.path.isdir(params.experiment_path): os.mkdir(params.experiment_path)
        
        logging.info("Building network graph on gpu: "+self.params.device)
        with tf.device(self.params.device):
            self.network.build(inputs = self.dataset.inputs, labels=self.dataset.labels, is_training=self.dataset.is_training)

        self.last_saver = None
        self.best_saver = None  
        self.begin_at_epoch = None

        self.train_writer = None
        self.eval_writer = None
        
        self.sess = None

    def initialize(self,restore_from=None):
        self.last_saver = tf.train.Saver() # will keep last 5 epochs
        self.best_saver = tf.train.Saver(max_to_keep=1)  # only keep 1 best checkpoint (best on eval)
        self.begin_at_epoch = 0

        self.best_eval_perf = -float("inf")

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

        self.sess.run(self.network.variable_init_op)

        if restore_from: self._restore(restore_from)

        self.train_writer = tf.summary.FileWriter(os.path.join(self.params.experiment_path, 'train_summaries'), self.sess.graph)
        self.eval_writer = tf.summary.FileWriter(os.path.join(self.params.experiment_path, 'eval_summaries'), self.sess.graph)
        
    def train(self, num_epochs=None, eval=True):
        if not self.sess:
            logging.info("Initializing model....")
            self.initialize()

        num_epochs = num_epochs if num_epochs else self.params.num_epochs
        for epoch in range(self.begin_at_epoch, self.begin_at_epoch + self.params.num_epochs):
            self._train(self.dataset.train_init_op, self.params.num_train_steps)
            self._eval(self.dataset.eval_init_op, self.params.num_eval_steps)

            self.save(epoch)

    def eval(self):
        self._eval(self.dataset.eval_init_op, self.params.num_eval_steps)
    
    def test(self):
        with tf.device(self.params.device):
            self.test_net = self.network.build(tf_dataset=self.dataset.test)
        self._eval(self.test_net,self.params.num_test_steps)
    
    def _restore(self, restore_from):
        logging.info("Restoring parameters from {}".format(restore_from))
        if os.path.isdir(restore_from):
            restore_from = tf.train.latest_checkpoint(restore_from)
            self.begin_at_epoch = int(restore_from.split('-')[-1])
        self.last_saver.restore(self.sess, restore_from)

    def save(self, epoch=0):
        logging.info("Finished Epoch {}/{}".format(epoch + 1, self.begin_at_epoch + self.params.num_epochs))
        self.last_saver.save(self.sess, os.path.join(self.params.experiment_path, 'last_weights', 'after-epoch'), global_step=epoch + 1)
        eval_perf = self.eval_metrics['accuracy'] if self.params.type == "classification" else -self.eval_metrics['mae']
        if eval_perf >= self.best_eval_perf:
            self.best_eval_perf = eval_perf
            best_save_path = self.best_saver.save(self.sess, os.path.join(self.params.experiment_path, 'best_weights', 'after-epoch'), global_step=epoch + 1)
            logging.info("- Found new best performance, saving in {}".format(best_save_path))

    def _train(self, data_init_op, num_steps):
        # Get relevant graph operations or nodes needed for training
        global_step = tf.train.get_global_step()

        # Load the training dataset into the pipeline
        self.sess.run(data_init_op)

        # Initialize the metrics local variables
        self.sess.run(self.network.metrics_init_op)

        t = trange(num_steps)
        for i in t:
            # Evaluate summaries for tensorboard only once in a while
            if i % self.params.save_summary_steps == 0:
                # Perform a mini-batch update
                _, _, loss_val, summ, global_step_val = self.sess.run([self.network.train_op, self.network.metrics_update_op, self.network.loss, self.network.summary_op, global_step])
                # Write summaries for tensorboard
                self.train_writer.add_summary(summ, global_step_val)
            else:
                _, _, loss_val = self.sess.run([self.network.train_op, self.network.metrics_update_op, self.network.loss])
            # Log the loss in the tqdm progress bar
            t.set_postfix(loss='{:05.3f}'.format(loss_val))
        
        self.log_metrics("Train", self.network.metrics)
    
    def _eval(self, data_init_op, num_steps):
        global_step = tf.train.get_global_step()

        # Load the evaluation dataset into the pipeline and initialize the metrics init op
        self.sess.run(data_init_op)
        self.sess.run(self.network.metrics_init_op)

        # compute metrics over the dataset
        for _ in range(num_steps):
            self.sess.run(self.network.metrics_update_op)

        # Get the values of the metrics
        self.log_metrics("Eval", self.network.metrics)

        # Add summaries manually to writer at global_step_val
        global_step_val = self.sess.run(global_step)
        for tag, val in self.eval_metrics.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            self.eval_writer.add_summary(summ, global_step_val)
    
    def log_metrics(self, m_type, metrics):
        metrics_values = {k: v[0] for k, v in metrics.items()}
        metrics_val = self.sess.run(metrics_values)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        if m_type == "Train":
            self.train_metrics = metrics_val
        else:
           self.eval_metrics = metrics_val
        logging.info(m_type + " metrics: " + metrics_string)