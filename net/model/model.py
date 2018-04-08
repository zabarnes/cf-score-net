"""Define the model."""

import tensorflow as tf
import logging
import numpy as np
from functools import reduce

VGG_PATH = "/data/CF_040718_eq/vgg19.npy"

class Model(object):
    def __init__(self, dataset, params, transfer = False):
        logging.info("Creating the model...")

        self.dataset = dataset
        self.params = params

        self.train_graph = Graph(is_training=True, inputs=dataset.train_inputs, transfer = transfer, params=params)
        self.eval_graph = Graph(is_training=False, inputs=dataset.eval_inputs, transfer = transfer, params=params)

        self.train_spec = self.train_graph.spec
        self.eval_spec = self.eval_graph.spec

class Graph(object):
    def __init__(self, is_training, inputs, transfer, params):
        self.params = params
        self.inputs = inputs
        self.is_training = is_training
        self.reuse = not is_training

        if transfer:
            self.build_fn = self.build_vgg_model
        else:
            self.build_fn = self.build_model

        if self.params.mode == "C":
            self.spec = self.build_graph()
        elif self.params.mode == "R":
            self.spec = self.build_regression_graph()

    def build_graph(self):
        """Model function defining the graph operations.

        Args:
            mode: (string) can be 'train' or 'eval'
            inputs: (dict) contains the inputs of the graph (features, labels...) this can be `tf.placeholder` or outputs of `tf.data`
            params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
            reuse: (bool) whether to reuse the weights

        Returns:
            model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
        """
        labels = self.inputs['labels']
        labels = tf.cast(labels, tf.int64)

        # -----------------------------------------------------------
        # MODEL: define the layers of the model
        with tf.variable_scope('model', reuse=self.reuse):
            # Compute the output distribution of the model and the predictions
            logits = self.build_fn()
            predictions = tf.argmax(logits, 1)

        # Define loss and accuracy
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

        # Define training step that minimizes the loss with the Adam optimizer
        if self.is_training:
            optimizer = tf.train.AdamOptimizer(self.params.learning_rate)
            global_step = tf.train.get_or_create_global_step()
            if self.params.use_batch_norm:
                # Add a dependency to update the moving mean and variance for batch normalization
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    train_op = optimizer.minimize(loss, global_step=global_step)
            else:
                train_op = optimizer.minimize(loss, global_step=global_step)

        # -----------------------------------------------------------
        # METRICS AND SUMMARIES
        # Metrics for evaluation using tf.metrics (average over whole dataset)
        with tf.variable_scope("metrics"):
            metrics = {
                'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
                'loss': tf.metrics.mean(loss),
                'auc': tf.metrics.auc(labels=labels,predictions=tf.nn.softmax(logits)[:,1])
            }

        # Group the update ops for the tf.metrics
        update_metrics_op = tf.group(*[op for _, op in metrics.values()])

        # Get the op to reset the local variables used in tf.metrics
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
        metrics_init_op = tf.variables_initializer(metric_variables)

        # Summaries for training
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.image('train_image', self.inputs['images'])

        # Add incorrectly labeled images
        mask = tf.not_equal(labels, predictions)

        # Add a different summary to know how they were misclassified
        for label in range(0, self.params.num_labels):
            mask_label = tf.logical_and(mask, tf.equal(predictions, label))
            incorrect_image_label = tf.boolean_mask(self.inputs['images'], mask_label)
            tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

        # -----------------------------------------------------------
        # MODEL SPECIFICATION
        # Create the model specification and return it
        # It contains nodes or operations in the graph that will be used for training and evaluation
        model_spec = self.inputs
        model_spec['variable_init_op'] = tf.global_variables_initializer()
        model_spec["predictions"] = predictions
        model_spec["labels"] = labels
        model_spec["probabilities"] = tf.nn.softmax(logits)
        model_spec['loss'] = loss
        model_spec['accuracy'] = accuracy
        model_spec['metrics_init_op'] = metrics_init_op
        model_spec['metrics'] = metrics
        model_spec['update_metrics'] = update_metrics_op
        model_spec['summary_op'] = tf.summary.merge_all()

        if self.is_training:
            model_spec['train_op'] = train_op

        return model_spec

    def build_regression_graph(self):
        """Model function defining the graph operations.

        Args:
            mode: (string) can be 'train' or 'eval'
            inputs: (dict) contains the inputs of the graph (features, labels...) this can be `tf.placeholder` or outputs of `tf.data`
            params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
            reuse: (bool) whether to reuse the weights

        Returns:
            model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
        """
        labels = self.inputs['labels']
        labels = tf.cast(labels, tf.int64)

        # -----------------------------------------------------------
        # MODEL: define the layers of the model
        with tf.variable_scope('model', reuse=self.reuse):
            # Compute the output distribution of the model and the predictions
            logits = self.build_fn()
            predictions = tf.squeeze(logits)
            #predictions = tf.argmax(logits, 1)

        # Define loss and accuracy
        #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss = tf.losses.mean_squared_error(labels = labels, predictions = predictions)
        difference = tf.reduce_mean(tf.abs(tf.subtract(tf.cast(labels,tf.float32),predictions)))

        # Define training step that minimizes the loss with the Adam optimizer
        if self.is_training:
            optimizer = tf.train.AdamOptimizer(self.params.learning_rate)
            global_step = tf.train.get_or_create_global_step()
            if self.params.use_batch_norm:
                # Add a dependency to update the moving mean and variance for batch normalization
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    train_op = optimizer.minimize(loss, global_step=global_step)
            else:
                train_op = optimizer.minimize(loss, global_step=global_step)

        # -----------------------------------------------------------
        # METRICS AND SUMMARIES
        # Metrics for evaluation using tf.metrics (average over whole dataset)
        with tf.variable_scope("metrics"):
            metrics = {
                'difference': tf.metrics.mean(difference),
                'loss': tf.metrics.mean(loss)
            }

        # Group the update ops for the tf.metrics
        update_metrics_op = tf.group(*[op for _, op in metrics.values()])

        # Get the op to reset the local variables used in tf.metrics
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
        metrics_init_op = tf.variables_initializer(metric_variables)

        # Summaries for training
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('difference', difference)
        tf.summary.image('train_image', self.inputs['images'])

        # -----------------------------------------------------------
        # MODEL SPECIFICATION
        # Create the model specification and return it
        # It contains nodes or operations in the graph that will be used for training and evaluation
        model_spec = self.inputs
        model_spec['variable_init_op'] = tf.global_variables_initializer()
        model_spec["difference"] = difference
        model_spec["labels"] = labels
        model_spec['loss'] = loss
        model_spec['metrics_init_op'] = metrics_init_op
        model_spec['metrics'] = metrics
        model_spec['update_metrics'] = update_metrics_op
        model_spec['summary_op'] = tf.summary.merge_all()

        if self.is_training:
            model_spec['train_op'] = train_op

        return model_spec

    def build_model(self):
        """Compute logits of the model (output distribution)

        Args:
            is_training: (bool) whether we are training or not
            inputs: (dict) contains the inputs of the graph (features, labels...)
                    this can be `tf.placeholder` or outputs of `tf.data`
            params: (Params) hyperparameters

        Returns:
            output: (tf.Tensor) output of the model
        """
        images = self.inputs['images']

        print(images.get_shape().as_list())

        assert images.get_shape().as_list() == [None, self.params.image_size, self.params.image_size, self.params.channel_dim]

        out = images
        # Define the number of channels of each convolution
        # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
        num_channels = self.params.num_channels
        bn_momentum = self.params.bn_momentum
        channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8, num_channels * 4]
        for i, c in enumerate(channels):
            with tf.variable_scope('block_{}'.format(i+1)):
                out = tf.layers.conv2d(out, c, 3, padding='same')
                if self.params.use_batch_norm:
                    out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=self.is_training)
                out = tf.nn.relu(out)
                out = tf.layers.max_pooling2d(out, 2, 2)

        print(out.get_shape().as_list())
        assert out.get_shape().as_list() == [None, 7, 7, num_channels * 4]

        out = tf.reshape(out, [-1, 7 * 7 * num_channels * 4])
        with tf.variable_scope('fc_1'):
            out = tf.layers.dense(out, num_channels * 8)
            if self.params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=self.is_training)
            out = tf.nn.relu(out)
        with tf.variable_scope('fc_2'):
            logits = tf.layers.dense(out, self.params.num_labels)

        return logits

    def build_vgg_model(self):
        self.data_dict = np.load(VGG_PATH, encoding='latin1').item()
        self.var_dict = {}
        self.trainable = self.is_training
        self.dropout = 0.5

        images = self.inputs['images']

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=images)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue,
            green,
            red,
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        if self.is_training:
            self.relu6 =  tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if self.is_training:
            self.relu7 =  tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, 4096, self.params.num_labels, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        
        return self.fc8

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count