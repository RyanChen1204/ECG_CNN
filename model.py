import tensorflow as tf
from cnn import getcnnfeature
from cnn import getcnnlogit


def loss(logits, label):
    """ loss function

    Args:
        logits : neural network output
        label : data label

    Returns:
        cost
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
    return cost


def prediction(logits, label):
    """ predict accuracy calculate

    Args:
        logits : neural network output
        label : data label

    Returns:
        accuracy
    """
    accuracy_single = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy_mean = tf.reduce_mean(tf.cast(accuracy_single, tf.float32))
    return accuracy_single, accuracy_mean


def inference(x, training, class_n):
    """ neural network

    Args:
        x : neural network input
        training : parameter normalization
        class_n : total label number

    Returns:
        neural network output
    """
    cnn_feature = getcnnfeature(x, training=training)
    logits = getcnnlogit(cnn_feature,class_n)
    return logits, cnn_feature


def _variable_on_cpu(name, shape, initializer,dtype = tf.float32):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
     shape: list of ints
     initializer: initializer for Variable
    Returns:
     Variable Tensor
    """
    # with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, wd, initializer,dtype = tf.float32):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    var = _variable_on_cpu(
        name,
        shape,
        initializer,
        dtype = dtype)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var