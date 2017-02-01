import tensorflow as tf
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def reduce_logsumexp(xs, axis=None, keep_dims=False):
  maxes = tf.reduce_max(xs, axis, keep_dims=True)
  xs -= maxes
  if keep_dims: return maxes + tf.log(tf.reduce_sum(tf.exp(xs), axis))
  else: return tf.squeeze(maxes, squeeze_dims=axis) + tf.log(tf.reduce_sum(tf.exp(xs), axis))

def dtanh(tensor):
    return 1.0 - tf.square(tf.tanh(tensor))

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    return tf.get_variable('weights', shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    return tf.get_variable('bias', shape, initializer=tf.constant_initializer(0.1))

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.variable_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.scalar_summary('mean', mean)
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.scalar_summary('stddev', stddev)
      tf.scalar_summary('max', tf.reduce_max(var))
      tf.scalar_summary('min', tf.reduce_min(var))
      tf.histogram_summary('histogram', var)
