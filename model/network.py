import tensorflow as tf
import tensorflow.contrib.layers as layers

import logging
        
class Network(object):
    def __init__(self, params):
        self.params = params

        # Defined handed by dataset when building
        self.inputs = None
        self.labels = None
        self.is_training = None

        self.reuse = False # Here for future with multi tower implementation

        # Constructed by building
        # self.logits = None
        # self.loss = None
        # self.predictions = None
        # self.metrics = None

        # # Operations
        # self.train_op = None
        # self.metrics_init_op = None
        # self.metrics_update_op = None
        # self.variable_init_op = None
        # self.summary_op = None

    def build(self, inputs, labels, is_training):
        # Specify graph params
        self.inputs, self.labels, self.is_training = inputs, labels, is_training[0]

        # Build model for prediction
        self.build_network()

        # Build loss
        self.build_loss()

        # Build train
        self.build_train_op()

        # Build metrics
        self.build_metrics()

    def build_network(self):
        with tf.variable_scope("model",reuse = self.reuse):

            assert self.inputs.get_shape().as_list() == [None, self.params.image_size, self.params.image_size, self.params.channel_dim]

            out = self.inputs
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

            assert out.get_shape().as_list() == [None, 7, 7, num_channels * 4]

            out = tf.reshape(out, [-1, 7 * 7 * num_channels * 4])
            with tf.variable_scope('fc_1'):
                out = tf.layers.dense(out, num_channels * 8)
                if self.params.use_batch_norm:
                    out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=self.is_training)
                out = tf.nn.relu(out)
            with tf.variable_scope('fc_2'):
                self.logits = tf.layers.dense(out, self.params.num_labels)

    def build_loss(self):
        self.predictions = tf.argmax(self.logits,1) if self.params.type == "classification" else tf.squeeze(self.logits,1)
        
        # Get loss from prediction
        if self.params.loss == "softmax":
            loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)
        elif self.params.loss == "l2":
            loss = tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions)
        elif self.params.loss == "ord":
            raise Exception("NotImplementedError")

        # Get loss from regularization
        regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = self.params.weight_decay * tf.reduce_sum(regs)

        # Set overall loss
        self.loss = loss + reg_loss
    
    def build_train_op(self):
        optimizer = self.get_optimizer()
        global_step = tf.train.get_or_create_global_step()

        grads_and_vars = optimizer.compute_gradients(self.loss, tf.trainable_variables())
        grads = [g for g, v in grads_and_vars]
        variables = [v for g, v in grads_and_vars]

        clipped_grads, global_norm = tf.clip_by_global_norm(grads, self.params.max_gradient_norm)

        if self.params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = optimizer.apply_gradients(zip(clipped_grads, variables), global_step = global_step, name = "apply_clipped_grads")
        else:
            self.train_op = optimizer.apply_gradients(zip(clipped_grads, variables), global_step = global_step, name = "apply_clipped_grads")

    def build_metrics(self):
        with tf.variable_scope("metrics"):
            self.metrics = {}
            self.metrics['loss'] = tf.metrics.mean(self.loss)
            if self.params.type == "classification":
                self.metrics["accuracy"] = tf.metrics.accuracy(labels=tf.cast(self.labels,tf.int32), predictions=tf.cast(self.predictions,tf.int32))
                self.metrics["auc"] = tf.metrics.auc(labels=self.labels,predictions=tf.nn.softmax(self.logits)[:,1])
            else:
                self.metrics['mae'] = tf.metrics.mean_absolute_error(labels=tf.cast(self.labels,tf.float32), predictions=self.predictions)
                self.metrics['rmse'] = tf.metrics.root_mean_squared_error(labels=tf.cast(self.labels,tf.float32), predictions=self.predictions)
                self.metrics['percentage_below_2'] = tf.metrics.percentage_below(tf.abs(tf.subtract(tf.cast(self.labels,tf.float32),self.predictions)), 2)

        self.metrics_update_op = tf.group(*[op for _, op in self.metrics.values()])
        self.metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
        self.metrics_init_op = tf.variables_initializer(self.metric_variables)

        tf.summary.scalar('loss', self.loss)
        tf.summary.image('train_image', self.inputs)
        if self.params.loss in ["softmax", "sigmoid"]:
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.labels,tf.int32), tf.cast(self.predictions,tf.int32)), tf.float32))
            tf.summary.scalar('accuracy',self.accuracy)
        else:
            self.difference = tf.reduce_mean(tf.abs(tf.subtract(tf.cast(self.labels,tf.float32), self.predictions)))
            tf.summary.scalar('difference',self.difference)

        # Add incorrectly labeled images
        mask = tf.not_equal(tf.cast(self.labels,tf.int32), tf.cast(self.predictions,tf.int32))

        # Add a different summary to know how they were misclassified
        for label in range(self.params.num_labels):
            mask_label = tf.logical_and(mask, tf.equal(tf.cast(self.predictions,tf.int32), label))
            incorrect_image = tf.boolean_mask(self.inputs, mask_label)
            tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image)
        
        self.variable_init_op = tf.global_variables_initializer()
        self.summary_op = tf.summary.merge_all()

    def get_optimizer(self):
        if self.params.optimizer == "adam":
            return tf.train.AdamOptimizer(self.params.learning_rate)   # Recommended lr of 1e-3
        elif self.params.optimizer == 'nesterov':   
            return tf.train.MomentumOptimizer(self.params.learning_rate, momentum = 0.9, use_nesterov = True)   # Recommended lr of 0.1, this is what was used in the resnet paper
        elif self.params.optimizer == 'rmsprop':
            return tf.train.RMSPropOptimizer(self.params.learning_rate)    # Recommended lr of 1e-2
        else:
            raise Exception("InvalidOptimizerError")

    def weight_decay(self):
        return layers.l2_regularizer(1.0)

    def weight_init(self):
        if self.params.initializer == "xavier":
            return layers.xavier_initializer()
        elif self.params.initializer == "fan_in":
            return layers.variance_scaling_initializer(mode='FAN_IN')
        elif self.params.initializer == "normal":
            return tf.truncated_normal_initializer(stddev=0.1)
        else:
            raise Exception("InvalidOptimizerError")

# AlexNet, but with reduced capacity
class AlexNet(Network):
    def __init__(self, params):
        super().__init__(params)

    def build_network(self):
        with tf.variable_scope("model",reuse = self.reuse):
            print("Input Shape:", self.inputs.shape)
            nn = layers.conv2d(self.inputs, num_outputs=64, kernel_size=7, stride=1, data_format='NHWC', padding='SAME', weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
            nn = layers.conv2d(nn, num_outputs=64, kernel_size=5, stride=1, data_format='NHWC', padding='SAME', weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
            nn = tf.nn.max_pool(nn, [1,2,2,1], strides=[1,2,2,1], padding='SAME', data_format='NHWC')
            nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = self.is_training)
            print(nn.shape)

            nn = layers.conv2d(nn, num_outputs=128, kernel_size=5, stride=1, data_format='NHWC', padding='SAME', weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
            nn = tf.nn.max_pool(nn, [1,2,2,1], strides=[1,2,2,1], padding='SAME', data_format='NHWC')
            nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = self.is_training)
            print(nn.shape)
            
            nn = layers.conv2d(nn, num_outputs=256, kernel_size=3, stride=1, data_format='NHWC', padding='SAME', weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
            nn = layers.conv2d(nn, num_outputs=256, kernel_size=3, stride=1, data_format='NHWC', padding='SAME', weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
            nn = tf.nn.max_pool(nn, [1,2,2,1], strides=[1,2,2,1], padding='SAME', data_format='NHWC')
            print(nn.shape)
            
            # Affine Layers
            nn = layers.flatten(nn)
            nn = layers.fully_connected(inputs = nn, num_outputs = 1024, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
            #nn = layers.dropout(nn, keep_prob = 0.5, is_training=is_training)
            nn = layers.fully_connected(inputs = nn, num_outputs = 1024, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
            #nn = layers.dropout(nn, keep_prob = 0.5, is_training=is_training)
            self.logits = layers.fully_connected(inputs = nn, num_outputs = self.params.num_labels, activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())

        assert (self.logits.get_shape().as_list() == [None, self.params.num_labels])

# A net using google inception modules
class GoogleNet(Network):
    def __init__(self, params):
        super().__init__(params)

    def inception_module(self, X, i, a, b, c, d, e, f):
        """
        a, b, c, d, e, f are integer inputs corresponding to the number of filters for various convolutions
        i is the number of the inception module. Used for namespacing
        """

        with tf.variable_scope("inception" + str(i)):
            conv3 = layers.conv2d(X, num_outputs=a, kernel_size=1, stride=1, data_format='NHWC', padding='SAME', scope = "Conv3")
            conv1 = layers.conv2d(X, num_outputs=b, kernel_size=1, stride=1, data_format='NHWC', padding='SAME', scope = "Conv1")
            conv2 = layers.conv2d(X, num_outputs=d, kernel_size=1, stride=1, data_format='NHWC', padding='SAME', scope = "Conv2")
            mp1 = tf.nn.max_pool(X, [1,3,3,1], strides=[1,1,1,1], padding='SAME', data_format='NHWC', name="max_pool")

            conv4 = layers.conv2d(conv1, num_outputs=c, kernel_size=3, stride=1, data_format='NHWC', padding='SAME', scope = "Conv4")
            conv5 = layers.conv2d(conv2, num_outputs=e, kernel_size=5, stride=1, data_format='NHWC', padding='SAME', scope = "Conv5")
            conv6 = layers.conv2d(mp1, num_outputs=f, kernel_size=1, stride=1, data_format='NHWC', padding='SAME', scope = "Conv6")
            out = tf.concat([conv3, conv4, conv5, conv6], axis=3, name="Concat")
            print(out.shape)
        return out

    def auxiliary_stem(self, X, i, is_training):
        print("stem: ", X.shape)
        with tf.variable_scope("gradient_helper_stem" + str(i)):
            nn = tf.nn.avg_pool(X, [1,5,5,1], strides=[1,3,3,1], padding='SAME', data_format='NHWC', name="avg_pool")
            nn = layers.conv2d(nn, num_outputs=128, kernel_size=1, stride=1, data_format='NHWC', padding='SAME', scope = "conv1")
            _, H1, W1, _ = nn.shape
            nn = tf.nn.avg_pool(nn, [1,H1,W1,1], strides=[1,1,1,1], padding='VALID', data_format='NHWC', name="avg_pool")
            nn = layers.dropout(nn, keep_prob = 0.7, is_training=is_training)
            nn = layers.flatten(nn)
            nn = layers.fully_connected(inputs = nn, num_outputs = self.params.num_labels, activation_fn = None, scope = "fc_out")
        return nn

    def build_network(self):
        with tf.variable_scope("model",reuse = self.reuse):
            # Stem Network
            nn = layers.conv2d(self.inputs, num_outputs=64, kernel_size=7, stride=1, data_format='NHWC', padding='SAME', scope = "Conv1")

            # Inception Layers. Params taken from GoogleNet Paper, cut in half
            nn = self.inception_module(nn, 1, 32, 48, 64, 8, 16, 16)
            nn = self.inception_module(nn, 2, 64, 64, 96, 16, 48, 32)
            nn = tf.nn.max_pool(nn, [1,3,3,1], strides=[1,2,2,1], padding='SAME', data_format='NHWC')
            nn = self.inception_module(nn, 3, 96, 48, 104, 8, 24, 32)
            incept3 = nn
            nn = self.inception_module(nn, 4, 80, 56, 112, 12, 32, 32)
            nn = self.inception_module(nn, 5, 64, 64, 64, 12, 32, 32)
            nn = tf.nn.max_pool(nn, [1,3,3,1], strides=[1,2,2,1], padding='SAME', data_format='NHWC')
            nn = self.inception_module(nn, 6, 55, 72, 72, 16, 32, 32)
            incept6 = nn
            nn = self.inception_module(nn, 7, 64, 80, 80, 8, 64, 64)
            nn = tf.nn.max_pool(nn, [1,3,3,1], strides=[1,2,2,1], padding='SAME', data_format='NHWC')
            nn = self.inception_module(nn, 8, 64, 80, 80, 16, 64, 64)
            nn = self.inception_module(nn, 9, 96, 96, 96, 24, 64, 64)

            # Classifier Output
            _, H1, W1, _ = nn.shape
            print("Before avg  pool: ", nn.shape)
            nn = tf.nn.avg_pool(nn, [1,H1,W1,1], strides=[1,1,1,1], padding='VALID', data_format='NHWC', name="avg_pool")  # Filter size is same as input size
            print("After avg pool: ", nn.shape)
            nn = layers.dropout(nn, keep_prob = 0.6, is_training=self.is_training)
            nn = layers.flatten(nn)
            self.logits = layers.fully_connected(inputs = nn, num_outputs = self.params.num_labels, activation_fn = None, scope = "fc_out")

            # Attach auxiliary output stems for helping grads propogate
            self.stem1_scores = self.auxiliary_stem(incept3, 1, self.is_training)
            self.stem2_scores = self.auxiliary_stem(incept6, 2, self.is_training)

        assert self.logits.get_shape().as_list() == [None, self.params.num_labels]

    def build_loss(self):
        self.predictions = tf.argmax(self.logits,1) if self.params.type == "classification" else tf.squeeze(self.logits,1)
        if self.params.loss == "softmax":
            l1 = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)
            l2 = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.stem1_scores)
            l3 = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.stem2_scores)
            self.loss = l1 + 0.3*l2 + 0.3*l3
        elif self.params.loss == "l2":
            l1 = tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions)
            l2 = tf.losses.mean_squared_error(labels=self.labels, predictions=tf.squeeze(self.stem1_scores,1))
            l3 = tf.losses.mean_squared_error(labels=self.labels, predictions=tf.squeeze(self.stem2_scores,1))
            self.loss = l1 + 0.3*l2 + 0.3*l3

# A 34 Layer Resnet
class ResNet(Network):
    def __init__(self, params):
        super().__init__(params)

    def ResLayer(self, x, filters, stride = 1, is_training = True, scope = "ResLayer"):
        with tf.variable_scope(scope):
            C = x.get_shape().as_list()[3]
            nn = layers.conv2d(x, num_outputs=filters, kernel_size=3, stride=stride, data_format='NHWC', padding='SAME', \
                activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
            nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn1", activation_fn = None)
            nn = tf.nn.relu(nn)

            nn = layers.conv2d(nn, num_outputs=filters, kernel_size=3, stride=1, data_format='NHWC', padding='SAME', \
                activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
            nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn2", activation_fn = None)

            if stride != 1:
                print("Projecting identity mapping to correct size")
                x = tf.nn.max_pool(x, [1,stride,stride,1], strides=[1,stride,stride,1], padding='SAME', data_format='NHWC') #previously used avg_pool
                x = layers.conv2d(x, num_outputs=filters, kernel_size=1, stride=1, data_format='NHWC', padding='SAME', \
                    activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())

            nn = x + nn     # Identity mapping plus residual connection
            nn = tf.nn.relu(nn)

            print("Image after " + scope + ":", nn.shape)
            return nn

    def BottleneckResLayer(self, x, filters1, filters2, stride = 1, is_training = True, scope = "BottleneckResLayer"):
        with tf.variable_scope(scope):
            C = x.get_shape().as_list()[3]
            nn = layers.conv2d(x, num_outputs=filters1, kernel_size=1, stride=1, data_format='NHWC', padding='SAME', \
                activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
            nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn1", activation_fn = None)
            nn = tf.nn.relu(nn)

            nn = layers.conv2d(x, num_outputs=filters1, kernel_size=3, stride=stride, data_format='NHWC', padding='SAME', \
                activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
            nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn1", activation_fn = None)
            nn = tf.nn.relu(nn)

            nn = layers.conv2d(nn, num_outputs=filters2, kernel_size=1, stride=1, data_format='NHWC', padding='SAME', \
                activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())

            nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn2", activation_fn = None)

            if stride != 1 or filters2 != C:
                print("Projecting identity mapping to correct size")
                x = tf.nn.avg_pool(x, [1,stride,stride,1], strides=[1,stride,stride,1], padding='SAME', data_format='NHWC')
                x = layers.conv2d(x, num_outputs=filters2, kernel_size=1, stride=1, data_format='NHWC', padding='SAME', \
                    activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())

            nn = x + nn     # Identity mapping plus residual connection
            nn = tf.nn.relu(nn)

            print("Image after " + scope + ":", nn.shape)
            return nn

    def WideResLayer(self, x, k, filters, stride = 1, is_training = True, scope = "ResLayer"):
        with tf.variable_scope(scope):
            C = x.get_shape().as_list()[3]

            nn = layers.batch_norm(x, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn1", activation_fn = None)
            nn = tf.nn.relu(nn)
            nn = layers.conv2d(nn, num_outputs=k*filters, kernel_size=3, stride=stride, data_format='NHWC', padding='SAME', \
                activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
            
            #nn = layers.dropout(nn, keep_prob = 0.8, is_training=is_training)
                        
            nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn2", activation_fn = None)
            nn = tf.nn.relu(nn)

            nn = layers.conv2d(nn, num_outputs=k*filters, kernel_size=3, stride=1, data_format='NHWC', padding='SAME', \
                activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())

            if stride != 1:
                print("Projecting identity mapping to correct size")
                x = tf.nn.max_pool(x, [1,stride,stride,1], strides=[1,stride,stride,1], padding='SAME', data_format='NHWC') #previously used avg_pool
                x = layers.conv2d(x, num_outputs=k*filters, kernel_size=1, stride=1, data_format='NHWC', padding='SAME', \
                    activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())

            nn = x + nn     # Identity mapping plus residual connection

            print("Image after " + scope + ":", nn.shape)
            return nn

# An 18 layer resnet
class ResNet18(ResNet):
    def __init__(self, params):
        super().__init__(params)

    def build_network(self):
        with tf.variable_scope("model",reuse = self.reuse):
            print("Input image: ", self.inputs.shape)
            nn = layers.conv2d(self.inputs, num_outputs=64, kernel_size=3, stride=1, data_format='NHWC', padding='SAME', \
                activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
            nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = self.is_training, scope = "bn1", activation_fn = None)
            nn = tf.nn.relu(nn)

            # Residual Layers
            nn = self.ResLayer(nn, 64, is_training = self.is_training, scope = "ResLayer1")
            nn = self.ResLayer(nn, 64, is_training = self.is_training, scope = "ResLayer2")
            nn = self.ResLayer(nn, 128, is_training = self.is_training, stride = 2, scope = "ResLayer4")
            nn = self.ResLayer(nn, 128, is_training = self.is_training, scope = "ResLayer5")
            nn = self.ResLayer(nn, 256, is_training = self.is_training, stride = 2, scope = "ResLayer8")
            nn = self.ResLayer(nn, 256, is_training = self.is_training, scope = "ResLayer9")
            nn = self.ResLayer(nn, 512, is_training = self.is_training, stride = 2, scope = "ResLayer14")
            nn = self.ResLayer(nn, 512, is_training = self.is_training, scope = "ResLayer15")

            # Output Stem
            _, H1, W1, _ = nn.shape
            nn = tf.nn.avg_pool(nn, [1,H1,W1,1], strides=[1,1,1,1], padding='VALID', data_format='NHWC', name="avg_pool")  # Filter size is same as input size
            nn = layers.flatten(nn)
            self.logits = layers.fully_connected(inputs = nn, num_outputs = self.params.num_labels, \
                activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())

        assert (self.logits.get_shape().as_list() == [None, self.params.num_labels])

# A 50 Layer Resnet
class DeepResNet(ResNet):
    def __init__(self, params):
        super().__init__(params)

    def build_network(self):
        with tf.variable_scope("model",reuse = self.reuse):
            print("Imput image: ", self.inputs.shape)
            nn = layers.conv2d(self.inputs, num_outputs=64, kernel_size=3, stride=1, data_format='NHWC', padding='SAME', \
                activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
            nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = self.is_training, scope = "bn1", activation_fn = None)
            nn = tf.nn.relu(nn)

            # Residual Layers
            nn = self.BottleneckResLayer(nn, 64, 256, is_training = self.is_training, scope = "ResLayer1")
            nn = self.BottleneckResLayer(nn, 64, 256, is_training = self.is_training, scope = "ResLayer2")
            nn = self.BottleneckResLayer(nn, 64, 256, is_training = self.is_training, scope = "ResLayer3")

            nn = self.BottleneckResLayer(nn, 128, 512, is_training = self.is_training, stride = 2, scope = "ResLayer4")
            nn = self.BottleneckResLayer(nn, 128, 512, is_training = self.is_training, scope = "ResLayer5")
            nn = self.BottleneckResLayer(nn, 128, 512, is_training = self.is_training, scope = "ResLayer6")
            nn = self.BottleneckResLayer(nn, 128, 512, is_training = self.is_training, scope = "ResLayer7")

            nn = self.BottleneckResLayer(nn, 256, 1024, is_training = self.is_training, stride = 2, scope = "ResLayer8")
            nn = self.BottleneckResLayer(nn, 256, 1024, is_training = self.is_training, scope = "ResLayer9")
            nn = self.BottleneckResLayer(nn, 256, 1024, is_training = self.is_training, scope = "ResLayer10")
            nn = self.BottleneckResLayer(nn, 256, 1024, is_training = self.is_training, scope = "ResLayer11")
            nn = self.BottleneckResLayer(nn, 256, 1024, is_training = self.is_training, scope = "ResLayer12")
            nn = self.BottleneckResLayer(nn, 256, 1024, is_training = self.is_training, scope = "ResLayer13")
            
            nn = self.BottleneckResLayer(nn, 512, 2048, is_training = self.is_training, stride = 2, scope = "ResLayer31")
            nn = self.BottleneckResLayer(nn, 512, 2048, is_training = self.is_training, scope = "ResLayer32")
            nn = self.BottleneckResLayer(nn, 512, 2048, is_training = self.is_training, scope = "ResLayer33")

            # Output Stem
            _, H1, W1, _ = nn.shape
            nn = tf.nn.avg_pool(nn, [1,H1,W1,1], strides=[1,1,1,1], padding='VALID', data_format='NHWC', name="avg_pool")  # Filter size is same as input size
            nn = layers.flatten(nn)
            self.logits = layers.fully_connected(inputs = nn, num_outputs = self.params.num_labels, activation_fn = None, \
                weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())

            assert (self.logits.get_shape().as_list() == [None, self.params.num_labels])

# WideResNet
class WideResNet32(ResNet):
    def __init__(self, params):
        super().__init__(params)

    def build_network(self):
        k = 10

        with tf.variable_scope("model",reuse = self.reuse):
            print("Input image: ", self.inputs.shape)
            nn = layers.conv2d(self.inputs, num_outputs=k*16, kernel_size=3, stride=2, data_format='NHWC', padding='SAME', \
                activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())

            # Residual Layers
            print("Pre WideResNet: ", nn.shape)
            nn = self.WideResLayer(nn, k, 16, is_training = self.is_training, scope = "ResLayer1")
            nn = self.WideResLayer(nn, k, 16, is_training = self.is_training, scope = "ResLayer2")
            nn = self.WideResLayer(nn, k, 16, is_training = self.is_training, scope = "ResLayer3")
            nn = self.WideResLayer(nn, k, 16, is_training = self.is_training, scope = "ResLayer4")
            nn = self.WideResLayer(nn, k, 32, is_training = self.is_training, stride = 2, scope = "ResLayer6")
            nn = self.WideResLayer(nn, k, 32, is_training = self.is_training, scope = "ResLayer7")
            nn = self.WideResLayer(nn, k, 32, is_training = self.is_training, scope = "ResLayer8")
            nn = self.WideResLayer(nn, k, 32, is_training = self.is_training, scope = "ResLayer10")
            nn = self.WideResLayer(nn, k, 64, is_training = self.is_training, stride = 2, scope = "ResLayer11")
            nn = self.WideResLayer(nn, k, 64, is_training = self.is_training, scope = "ResLayer12")
            nn = self.WideResLayer(nn, k, 64, is_training = self.is_training, scope = "ResLayer13")
            nn = self.WideResLayer(nn, k, 64, is_training = self.is_training, scope = "ResLayer15")

            # Output Stem
            _, H1, W1, _ = nn.shape
            nn = tf.nn.avg_pool(nn, [1,H1,W1,1], strides=[1,1,1,1], padding='VALID', data_format='NHWC', name="avg_pool")  # Filter size is same as input size
            nn = layers.flatten(nn)
            self.logits = layers.fully_connected(inputs = nn, num_outputs = self.params.num_labels, \
                activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())

        assert (self.logits.get_shape().as_list() == [None, self.params.num_labels])