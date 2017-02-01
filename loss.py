
import tensorflow as tf
from nn_utils import *
#from tensorflow.contrib import losses
import tensorflow.contrib.losses as losses

def discriminator_loss(D1, D2):
    '''Loss for the discriminator network
    Args:
        D1: logits computed with a discriminator networks from real images
        D2: logits computed with a discriminator networks from generated images
    Returns:
        Cross entropy loss, positive samples have implicit labels 1, negative 0s
    '''
    with tf.name_scope('discriminator_loss'):
        D_loss =  tf.add(losses.sigmoid_cross_entropy(D1, tf.ones(tf.shape(D1))),
            losses.sigmoid_cross_entropy(D2, tf.zeros(tf.shape(D1))),
            name='D_loss')
    return D_loss

def generator_loss(D2):
    '''Loss for the generator. Maximize probability of generating images that
    discriminator cannot differentiate.
    Returns:
        see the paper
    '''
    with tf.name_scope('generator_loss'):
        G_loss = tf.identity(losses.sigmoid_cross_entropy(D2, tf.ones(tf.shape(D2))),
                name='G_loss')
    return G_loss

def softmax_cross_entropy(logits, labels, reduce=True):
    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    if reduce:
        softmax_cross_entropy = tf.reduce_mean(softmax_cross_entropy, axis=0, name='cross_entropy')
    else:
        softmax_cross_entropy = tf.identity(softmax_cross_entropy, name='cross_entropy_batched')
    #tf.summary.scalar('cross_entropy', softmax_cross_entropy)
    return softmax_cross_entropy

def accuracy_with_logits(logits, labels, weights=None):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy_ = tf.cast(correct_prediction, "float")

    if weights is not None:
        accuracy_ = tf.mul(accuracy_, weights)
        accuracy = tf.reduce_sum(accuracy_, name='accuracy')
    else:
        accuracy = tf.reduce_mean(accuracy_, name='accuracy')

    #tf.summary.scalar('accuracy', accuracy)
    return accuracy

def fraction_above_threshold(logits, threshold):
    above_threshold = logits > threshold
    f_a_t = tf.reduce_mean(tf.cast(above_threshold, "float"), name='fraction_above_threshold')
    return f_a_t

def fraction_above_zero(logits):
    return fraction_above_threshold(logits, 0.0)

def elbo_loss(x, x_rec, z_mean, z_log_sigma_sq):
    reconstr_loss = reconstruction_loss(x, x_rec)
    latent_loss = latent_loss(z_mean, z_log_sigma_sq)
    elbo_loss = tf.reduce_mean(reconstr_loss + latent_loss, name='elbo_loss')
    return tf.identity(elbo_loss, name='elbo_loss')

def reconstruction_loss(x, x_rec, reduce=True):
    reconstr_loss = -tf.reduce_sum(x * tf.log(1e-10 + x_rec)
                           + (1 - x) * tf.log(1e-10 + 1 - x_rec), 1)
    if reduce:
        reconstr_loss = tf.reduce_mean(reconstr_loss)
    return tf.identity(reconstr_loss, name='reconstruction_loss')

def latent_loss(z_mean, z_log_sigma_sq, reduce=True):
    latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq
                                           - tf.square(z_mean)
                                           - tf.exp(z_log_sigma_sq), 1)
    if reduce:
        latent_loss = tf.reduce_mean(latent_loss)
    return tf.identity(latent_loss, name='latent_loss')
