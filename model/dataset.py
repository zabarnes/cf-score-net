import logging
import os
import random
import math

import tensorflow as tf
import pandas as pd
import numpy as np

from model.utils import set_logger

DATA_DIR = "data"
LABEL_FNAME = "labels.csv"

class Dataset(object):
    def __init__(self, params):
        logging.info("Initializing dataset ...")
        self.dataset_path = params.dataset_path
        self.params = params

        if not os.path.isdir(params.experiment_path): os.mkdir(params.experiment_path)
        tf.set_random_seed(100)
        set_logger(os.path.join(params.experiment_path, 'experiment.log'))

        #Get the file paths for data
        self.get_data_path()

        #Split the train data and verify all data contents
        self.read_and_verify_data()

        #Build tf datasets for each set of inputs
        self.train_dataset = self.build_tf_dataset(self.train_filenames, self.train_labels, is_training=True)
        self.eval_dataset = self.build_tf_dataset(self.eval_filenames, self.eval_labels, is_training=False)
        self.test_dataset = self.build_tf_dataset(self.test_filenames, self.test_labels, is_training=False)

        #Build singular dataset iterator
        self.dataset_iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types,self.train_dataset.output_shapes)
        self.inputs, self.labels, self.is_training = self.dataset_iterator.get_next()

        #Build init ops for train, eval, and test datasets
        self.train_init_op = self.dataset_iterator.make_initializer(self.train_dataset)
        self.eval_init_op = self.dataset_iterator.make_initializer(self.eval_dataset)
        self.test_init_op = self.dataset_iterator.make_initializer(self.eval_dataset)

    def get_data_path(self):
        self.train_label_path = os.path.join(self.dataset_path, "train_"+LABEL_FNAME)
        self.test_label_path = os.path.join(self.dataset_path, "test_"+LABEL_FNAME)

        assert os.path.isfile(self.train_label_path) 
        assert os.path.isfile(self.test_label_path)

        self.data_path = os.path.join(self.dataset_path, DATA_DIR)

        assert os.path.isdir(self.data_path) 

        logging.info("Found data at {}".format(self.data_path))
        logging.info("Found train labels at {} and test labels at {}".format(self.train_label_path, self.test_label_path))

    def read_and_verify_data(self):
        total_datafile = pd.read_csv(self.train_label_path)
        test_data = pd.read_csv(self.test_label_path)

        train_mask = np.random.rand(len(total_datafile)) <= self.params.train_split
        train_data = total_datafile[train_mask]
        eval_data = total_datafile[~train_mask]

        logging.info("Verifying dataset ...")
        for f in total_datafile.filenames.append(test_data.filenames):
            fp = os.path.join(self.data_path,f)
            assert os.path.isfile(fp), "{} isnt found".format(fp)

        self.params.train_size = len(train_data)
        self.params.eval_size = len(eval_data)
        self.params.test_size = len(test_data)
        logging.info("Training Examples: {}, Evaluation Examples: {}, Testing Examples: {}".format(self.params.train_size,self.params.eval_size,self.params.test_size))

        self.params.num_train_steps = (self.params.train_size + self.params.batch_size - 1) // self.params.batch_size
        self.params.num_eval_steps = (self.params.eval_size + self.params.batch_size - 1) // self.params.batch_size
        self.params.num_test_steps = (self.params.test_size + self.params.batch_size - 1) // self.params.batch_size

        self.train_filenames = list(self.data_path + "/" + train_data.filenames)
        self.eval_filenames = list(self.data_path + "/" + eval_data.filenames)
        self.test_filenames = list(self.data_path + "/" + test_data.filenames)

        self.train_labels = list(train_data[self.params.labels_field])
        self.eval_labels = list(eval_data[self.params.labels_field])
        self.test_labels = list(test_data[self.params.labels_field])

    def build_tf_dataset(self, filenames, labels, is_training):
        # Build a tf dataset on the list of filenames and labels
        tf_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels), tf.constant([is_training]*len(labels))))

        # Training input pipeline
        if is_training:
            tf_dataset = (tf_dataset.shuffle(self.params.train_size)  
                                    .map(self._parse_data, num_parallel_calls=self.params.num_parallel_calls)
                                    .map(self._preprocess_data, num_parallel_calls=self.params.num_parallel_calls)
                                    .batch(self.params.batch_size)
                                    .prefetch(1))
        # Eval and test pipeline
        else:
            tf_dataset = (tf_dataset.map(self._parse_data)
                                    .batch(self.params.batch_size)
                                    .prefetch(1))

        # Create dataset
        return tf_dataset

    def _parse_data(self, filename, label, is_training):
        image_string = tf.read_file(filename)

        # Don't use tf.image.decode_image, or the output shape will be undefined
        image = tf.image.decode_png(image_string, channels=3)

        if self.params.channel_dim == 1:
            image = tf.image.rgb_to_grayscale(image)

        # This will convert to float values in [0, 1)
        image = tf.image.convert_image_dtype(image, tf.float32)

        image = tf.image.resize_images(image, [self.params.image_size, self.params.image_size])

        return image, label, is_training

    def _preprocess_data(self, image, label, is_training):
        image = tf.contrib.image.rotate(image, tf.random_uniform([1], maxval=.25*math.pi))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=.2)
        image = tf.image.random_contrast(image,lower=0.85, upper=1.15)
        image = tf.random_crop(image, [self.params.crop_size, self.params.crop_size, self.params.channel_dim])

        image = tf.image.resize_images(image, [self.params.image_size, self.params.image_size])
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label, is_training