"""Create the input data pipeline using `tf.data`"""
import logging
import os
import random
import math

import tensorflow as tf

from model.utils import read_csv

class Dataset(object):
    def __init__(self, data_dir, label_path, params):
        self.params = params
        logging.info("Initializing the dataset...")
        self.labels = list(read_csv(label_path).items())

        self.size = len(self.labels)
        self.train_size = math.floor(params.train_split * self.size)
        if params.debug: self.train_size = 1
        self.eval_size = self.size - self.train_size

        self.params.train_size = self.train_size
        self.params.eval_size = self.eval_size

        self.train_data, self.eval_data = self.split_dataset()
        logging.info("Training Examples: {}, Evaluation Examples:{}".format(len(self.train_data),len(self.eval_data)))

        self.train_inputs = self.get_iterator(is_training = True, data=self.train_data)
        self.eval_inputs = self.get_iterator(is_training = False, data=self.eval_data)

    def split_dataset(self):
        random.shuffle(self.labels)
        return self.labels[:self.train_size], self.labels[self.train_size:]

    def _parse_function(self, filename, label, size):
        """
        Obtain the image from the filename (for both training and validation).
        The following operations are applied:
            - Decode the image from jpeg format
            - Convert to float and to range [0, 1]
        """

        image_string = tf.read_file(filename)

        # Don't use tf.image.decode_image, or the output shape will be undefined
        image_decoded = tf.image.decode_png(image_string, channels=3)

        if self.params.channel_dim == 1:
            image_decoded = tf.image.rgb_to_grayscale(image_decoded)

        # This will convert to float values in [0, 1)
        image = tf.image.convert_image_dtype(image_decoded, tf.float32)

        resized_image = tf.image.resize_images(image, [size, size])

        return resized_image, label

    def train_preprocess(self, image, label, use_random_flip):
        """
        Image preprocessing for training.
        Apply the following data augmentation
        """

        image = tf.contrib.image.rotate(image, tf.random_uniform([1], maxval=.25*math.pi))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=.2)
        image = tf.image.random_contrast(image,lower=0.85, upper=1.15)
        image = tf.random_crop(image, [self.params.crop_size, self.params.crop_size, self.params.channel_dim])

        image = tf.image.resize_images(image, [self.params.image_size, self.params.image_size])
        # Make sure the image is still in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label

    def get_iterator(self, is_training, data):
        """
        Args:
            is_training: (bool) whether to use the train or test pipeline.
                        At training, we shuffle the data and have multiple epochs
            data: data to train on 
            params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
        """
        filenames, labels = map(list, zip(*data))

        # Create a function to parse images
        parse_fn = lambda f, l: self._parse_function(f, l, self.params.image_size)

        # Create a function for training
        train_fn = lambda f, l: self.train_preprocess(f, l, self.params.use_random_flip)

        # Create a Dataset serving batches of images and labels
        if is_training:
            dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                .shuffle(self.train_size)  # whole dataset into the buffer ensures good shuffling
                .map(parse_fn, num_parallel_calls=self.params.num_parallel_calls)
                .map(train_fn, num_parallel_calls=self.params.num_parallel_calls)
                .batch(self.params.batch_size)
                .prefetch(1)  # make sure you always have one batch ready to serve
            )
        else:
            dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                .map(parse_fn)
                .batch(self.params.batch_size)
                .prefetch(1)  # make sure you always have one batch ready to serve
            )

        # Create reinitializable iterator from dataset
        iterator = dataset.make_initializable_iterator()
        images, labels = iterator.get_next()
        iterator_init_op = iterator.initializer

        return {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}