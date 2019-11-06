import tensorflow as tf
import model as model


def conv_layer(indata, ksize, padding, training, name, dilate=1, strides=None, bias_term=False, active=True,
               BN=True, active_function='relu', wd=None):
    """A convolutional layer

    Args:
        indata: A input 4D-Tensor of shape [batch_size, Height, Width, Channel].
        ksize: A length 4 list.
        padding: A String from: "SAME","VALID"
        training: Scalar Tensor of type boolean, indicate if in training or not.
        name: A String give the name of this layer, other variables and options created in this layer will have this name as prefix.
        dilate (int, optional): Defaults to 1. Dilation of the width.
        strides (list, optional): Defaults to [1, 1, 1, 1]. A list of length 4.
        bias_term (bool, optional): Defaults to False. If True, a bais Tensor is added.
        active (bool, optional): Defaults to True. If True, output is activated by a activation function.
        BN (bool, optional): Defaults to True. If True, batch normalization will be applied. 
        active_function (str, optional): Defaults to 'relu'. A String from 'relu','sigmoid','tanh'.
        wd: weight decay, if None no weight decay will be added.

    Returns:
        conv_out: A output 4D-Tensor.
    """
    if strides is None:
        strides = [1, 1, 1, 1]
    with tf.variable_scope(name):
        W = model._variable_with_weight_decay("weights",
                                        shape=ksize,
                                        wd=wd,
                                        initializer=tf.contrib.layers.xavier_initializer(uniform=False, ))
        conv_out = tf.nn.conv2d(
            indata, W, strides=strides, padding=padding, name=name)
    if BN:
        with tf.variable_scope(name + '_bn') as scope:
            # conv_out = batchnorm(conv_out,scope=scope,training = training)
            conv_out = simple_global_bn(conv_out, name=name + '_bn')
            # conv_out = tf.layers.batch_normalization(conv_out,axis = -1,training = training,name = 'bn')
    if active:
        if active_function == 'relu':
            with tf.variable_scope(name + '_relu'):
                conv_out = tf.nn.relu(conv_out, name='relu')
        elif active_function == 'sigmoid':
            with tf.variable_scope(name + '_sigmoid'):
                conv_out = tf.sigmoid(conv_out, name='sigmoid')
        elif active_function == 'tanh':
            with tf.variable_scope(name + '_tanh'):
                conv_out = tf.tanh(conv_out, name='tanh')
    return conv_out


def batchnorm(inp, scope, training, decay=0.99, epsilon=1e-5):
    """Applied batch normalization on the last axis of the tensor.

    Args:
        inp: A input Tensor
        scope: A string or tf.VariableScope.
        training (Boolean)): A scalar boolean tensor.
        decay (float, optional): Defaults to 0.99. The mean renew as follow: mean = pop_mean * (1- decay) + decay * old_mean
        epsilon (float, optional): Defaults to 1e-5. A small float number to avoid dividing by 0.

    Returns:
        The normalized, scaled, offset tensor.
    """

    with tf.variable_scope(scope):
        size = inp.get_shape().as_list()[-1]
        scale = tf.get_variable(
            'scale', shape=[size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', shape=[size])

        pop_mean = tf.get_variable(
            'pop_mean', shape=[size], initializer=tf.zeros_initializer(), trainable=False)
        pop_var = tf.get_variable(
            'pop_var', shape=[size], initializer=tf.ones_initializer(), trainable=False)
        batch_mean, batch_var = tf.nn.moments(inp, [0, 1, 2])

        train_mean_op = tf.assign(
            pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(
            pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(inp, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(inp, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(training, batch_statistics, population_statistics)


def simple_global_bn(inp, name):
    """Global batch normalization
    This tensor is nomalized by the global mean of the input tensor along the last axis.

    Args:
        inp : A 4D-Tensor.
        name (str): Name of the operation.

    Returns:
        global batch normalized tensor.
    """

    ksize = inp.get_shape().as_list()
    ksize = [ksize[-1]]
    mean, variance = tf.nn.moments(inp, [0, 1, 2], name=name + '_moments')
    scale = model._variable_on_cpu(name + "_scale",
                             shape=ksize,
                             initializer=tf.contrib.layers.variance_scaling_initializer())
    offset = model._variable_on_cpu(name + "_offset",
                              shape=ksize,
                              initializer=tf.contrib.layers.variance_scaling_initializer())
    return tf.nn.batch_normalization(inp, mean=mean, variance=variance, scale=scale, offset=offset,
                                     variance_epsilon=1e-5)


def residual_layer(indata, out_channel, training, i_bn=False):
    """An inplementation of the residual layer from https://arxiv.org/abs/1512.03385

    Args:
        indata: A 4-D Tensor
        out_channel (Int): The number of out channel
        training (Boolean): 0-D Boolean Tensor indicate if it's in training.
        i_bn (bool, optional): Defaults to False. If the identity layer being batch nomalized.

    Returns:
        relu_out: A 4-D Tensor of shape [batch_size, Height, Weight, out_channel]
    """

    fea_shape = indata.get_shape().as_list()
    in_channel = fea_shape[-1]
    with tf.variable_scope('branch1'):
        indata_cp = conv_layer(indata, ksize=[1, 3, in_channel, out_channel], padding='SAME', training=training,
                               name='conv1', BN=i_bn, active=False)
    with tf.variable_scope('branch2'):
        conv_out1 = conv_layer(indata, ksize=[1, 3, in_channel, out_channel], padding='SAME', training=training,
                               name='conv2a', bias_term=False)
        conv_out2 = conv_layer(conv_out1, ksize=[1, 5, out_channel, out_channel], padding='SAME', training=training,
                               name='conv2b', bias_term=False)
        conv_out3 = conv_layer(conv_out2, ksize=[1, 3, out_channel, out_channel], padding='SAME', training=training,
                               name='conv2c', bias_term=False, active=False)
    with tf.variable_scope('plus'):
        relu_out = tf.nn.relu(indata_cp + conv_out3, name='final_relu')
    return relu_out


def getcnnfeature(signal, training):
    """Compute the CNN feature given the signal input.  

    Args:
        signal (Float): A 2D-Tensor of shape [batch_size,max_time]
        training (Boolean): A 0-D Boolean Tensor indicate if it's in training.      

    Returns:
        cnn_fea: A 3D-Tensor of shape [batch_size, max_time, channel]
    """
    signal_shape = signal.get_shape().as_list()
    signal = tf.reshape(signal, [signal_shape[0], 1, signal_shape[1], 1])
    with tf.variable_scope('res_layer1'):
        res1 = residual_layer(signal, out_channel=256,
                              training=training, i_bn=True)
    with tf.variable_scope('res_layer2'):
        res2 = residual_layer(res1, out_channel=256, training=training)
    with tf.variable_scope('res_layer3'):
        res3 = residual_layer(res2, out_channel=256, training=training)
    feashape = res3.get_shape().as_list()
    fea = tf.reshape(res3, [feashape[0], feashape[2],
                            feashape[3]], name='fea_rs')
    return fea


def getcnnlogit(fea, class_n):
    """Get the logits from CNN feature.

    Args:
        fea (Float): A 3D-Tensor of shape [batch_size,max_time,channel]
        outnum (int, optional): Defaults to 5. Output class number, A,G,C,T,<ctc-blank>.

    Returns:
        A 3D-Tensor of shape [batch_size,max_time,outnum]
    """

    to_class_n = 1024
    feashape = fea.get_shape().as_list()
    fea_reshape = tf.reshape(fea, [feashape[0], feashape[1] * feashape[2]])
    w_01 = tf.get_variable("fully_weights_01", shape=[
        feashape[1] * feashape[2], to_class_n], initializer=tf.contrib.layers.xavier_initializer())
    b_01 = tf.get_variable("fully_bias_01", shape=[
        to_class_n], initializer=tf.contrib.layers.xavier_initializer())

    w_02 = tf.get_variable("fully_weights_02", shape=[
        to_class_n, class_n], initializer=tf.contrib.layers.xavier_initializer())
    b_02 = tf.get_variable("fully_bias_02", shape=[
        class_n], initializer=tf.contrib.layers.xavier_initializer())

    fully_output_01 = tf.nn.bias_add(tf.matmul(fea_reshape, w_01), b_01, name='fully_output_01')
    fully_output_02 = tf.nn.bias_add(tf.matmul(fully_output_01, w_02), b_02, name='fully_output_02')

    return fully_output_02
