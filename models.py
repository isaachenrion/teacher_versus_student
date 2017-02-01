from constants import *
import tensorflow as tf
from nn_utils import bias_variable, weight_variable
from loss import *
import numpy as np
import os
from utils import save_visualizations
from datasets import ProtoLabelledDataset

from scipy.misc import imsave

class Model(object):
    def __init__(self, paradigm):
        self.paradigm = paradigm
        self.global_step = tf.contrib.framework.get_or_create_global_step()

    def get_feed_dict(self, *args):
        raise NotImplementedError

    def update_params(self, *args):
        raise NotImplementedError

    def get_global_step(self):
        return self.sess.run(self.global_step)

    def train(self, total_loss, global_step, optimizer, var_list=None):
        if var_list is None:
            var_list = tf.trainable_variables()
        with tf.control_dependencies([total_loss]):
            grads = optimizer.compute_gradients(total_loss, var_list=var_list)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        for var in var_list:
            tf.summary.histogram(var.op.name, var)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            0.99, global_step)
        variables_averages_op = variable_averages.apply(var_list)
        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')
        return train_op

    def evaluate(self, *args):
        raise NotImplementedError

    def setup_evaluations(self, *args):
        raise NotImplementedError


class Classifier(Model):
    def __init__(self,
                input_dim=None,
                label_dim=None,
                optimizer=None,
                input_tensor=None,
                label_tensor=None,
                classify_fn=None,
                sess=None):
        super(Classifier, self).__init__(paradigm='classifier')

        self.classify_fn = classify_fn
        # checks
        assert optimizer is not None
        self.optimizer = optimizer

        self.input_dim = input_dim
        self.label_dim = label_dim
        self.set_model_tensors()

        self.logits = self.classify(self.inputs)

        self.set_loss_op()
        self.set_train_op()
        #self.setup_evaluations()
        self.set_summaries()
        self.check_op = tf.add_check_numerics_ops()

        if sess is not None:
            self.sess = sess # this classifier is not the chief!
        else:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def set_model_tensors(self):
        self.inputs = tf.placeholder(tf.float32, [None, self.input_dim])
        self.labels = tf.placeholder(tf.float32, [None, self.label_dim])

    def set_logits(self):
        self.logits = self.classify(self.inputs)
        self.logits = self.classify(self.inputs)

    def set_train_op(self):
        self.train_op = self.train(self.loss_op, self.global_step, self.optimizer)

    def set_loss_op(self):
        self.loss_op = softmax_cross_entropy(logits=self.logits, labels=self.labels)

    def set_summaries(self):
        tf.summary.scalar('loss', self.loss_op)
        tf.summary.scalar('accuracy', accuracy_with_logits(self.logits, self.labels))
        self.summary_op = tf.summary.merge_all()

    def classify(self, inputs):
        if self.classify_fn is None:
            raise NotImplementedError
        else: return self.classify_fn(inputs)

    def get_feed_dict(self, inputs, labels):
        inputs = np.reshape(inputs, [-1] + self.input_shape)
        feed_dict={self.inputs: inputs, self.labels: labels}
        return feed_dict

    def update_params(self, inputs, labels):
        feed_dict = self.get_feed_dict(inputs, labels)
        loss_value, _ = self.sess.run([self.loss_op, self.train_op], feed_dict)
        return loss_value

    def evaluate(self, inputs, labels):
        feed_dict=self.get_feed_dict(inputs, labels)
        loss_value, summary_str = self.sess.run([self.loss_op, self.summary_op], feed_dict=feed_dict)
        return loss_value, summary_str

    def setup_evaluations(self):
        evaluation_list = tf.get_collection('evaluation')
        tf.add_to_collection('evaluation', self.loss_op)
        self.accuracy = accuracy_with_logits(logits=self.logits, labels=self.labels)
        tf.add_to_collection('evaluation', self.accuracy)
        evaluation_list = tf.get_collection('evaluation')
        self.evaluation_dict = {evaluation.name: evaluation for evaluation in evaluation_list}
        assert len(self.evaluation_dict.keys()) > 0
        self.evaluations_have_been_setup = True


class FullyConnectedClassifier(Classifier):
    def __init__(self, input_shape, **kwargs):
        self.input_shape = input_shape
        input_dim = np.prod(input_shape)
        super(FullyConnectedClassifier, self).__init__(input_dim=input_dim, **kwargs)

    def set_model_tensors(self):
        self.inputs = tf.placeholder(tf.float32, [None] + self.input_shape, name='inputs')
        self.labels = tf.placeholder(tf.float32, [None, self.label_dim], name='labels')

    def classify(self, inputs):
        with tf.variable_scope('classifier') as s:
            reshape = tf.reshape(inputs, [-1, self.input_dim])
            with tf.variable_scope('fc1') as scope:
                weights = weight_variable([self.input_dim, FC1])
                biases = bias_variable([FC1])
                pre_act = tf.add(tf.matmul(reshape, weights), biases)
                fc1 = tf.nn.relu(pre_act, name=scope.name)
            with tf.variable_scope('fc2') as scope:
                weights = weight_variable([FC1, FC2])
                biases = bias_variable([FC2])
                pre_act = tf.add(tf.matmul(fc1, weights), biases)
                fc2 = tf.nn.relu(pre_act, name=scope.name)
            with tf.variable_scope('fc3') as scope:
                weights = weight_variable([FC2, self.label_dim])
                biases = bias_variable([self.label_dim])
                pre_act = tf.add(tf.matmul(fc2, weights), biases)
                fc3 = tf.identity(pre_act, name=scope.name)
        return fc3


class ConvnetClassifier(Classifier):
    def __init__(self, input_shape, **kwargs):
        self.input_shape = input_shape
        super(ConvnetClassifier, self).__init__(**kwargs)

    def set_model_tensors(self):
        self.inputs = tf.placeholder(tf.float32, [None] + self.input_shape)
        self.labels = tf.placeholder(tf.float32, [None, self.label_dim])

    def classify(self, inputs):
        reshape = inputs
        with tf.variable_scope('conv1') as scope:
            kernel = weight_variable(CONV1)
            conv = tf.nn.conv2d(reshape, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable(CONV1[3])
            pre_act = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_act, name=scope.name)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

        with tf.variable_scope('conv2') as scope:
            kernel = weight_variable(CONV2)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable(CONV2[3])
            pre_act = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_act, name=scope.name)

        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        with tf.variable_scope('fc3') as scope:
            dim = np.prod(pool2.get_shape()[1:].as_list())
            reshape = tf.reshape(pool2, [-1, dim])
            weights = weight_variable([dim, self.label_dim])
            biases = bias_variable([self.label_dim])
            pre_act = tf.add(tf.matmul(reshape, weights), biases)
            fc3 = tf.identity(pre_act, name=scope.name)

        return fc3


class SafetyNetClassifier(Classifier):
    def __init__(self, cost_per_query, safety_net_optimizer, **kwargs):
        super(SafetyNetClassifier, self).__init__(**kwargs)

        #with tf.variable_scope('main_classifier') as self.classifier_scope:

            #self.logits, self.features = self.classify(self.inputs, features=True)
            #feature_dim = self.features.get_shape().as_list()[-1]

        with tf.variable_scope('safety_net') as self.safety_net_scope:
            self.error_threshold = tf.constant(0.1)
            self.cost_per_query = tf.constant(query_cost, tf.float32, name='cost_per_query')
            self.safety_net = Classifier(input_dim=input_dim,
                                        label_dim=2,
                                        optimizer=safety_net_optimizer,
                                        classify_fn=self.compute_safety,
                                        loss_fn=tf.nn.sparse_softmax_cross_entropy_with_logits,
                                        sess=self.sess)
            self.clf_errors = self.get_clf_errors()
            safety_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.safety_net_scope)
            self.train_safety_net = self.safety_net.train(self.safety_net.loss_op, self.global_step, self.safety_net.optimizer, var_list=safety_net_params)

        self.classifier_data = ProtoLabelledDataset(batch_size=CLF_BATCH_SIZE)
        self.sn_data = ProtoLabelledDataset(batch_size=SN_BATCH_SIZE)

        classifier_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.classifier_scope)
        self.clf_train_op = self.train(self.loss_op, self.global_step, self.optimizer, var_list=classifier_params)


    def update_params(self, inputs, labels):
        # split the data in 2
        clf_inputs, sn_inputs = np.split(inputs, 2)
        clf_labels, sn_labels = np.split(labels, 2)

        # add 1/2 to safety net data
        self.classifier_data.add_data(sn_inputs, sn_labels)

        # now use the safety net on the other 1/2 to determine which of these
        # should be queried, and add them to the clf data
        subset_x, subset_y = self.subset_selection(inputs, labels)
        self.sn_data.add_data(subset_x, subset_y)

        # fetch minibatch from clf data and train
        inputs, labels = classifier_data.get_batch()
        feed_dict = self.get_feed_dict(inputs, labels)
        _, l_clf = self.sess.run([self.clf_train_op, self.loss_op], feed_dict=feed_dict)

        # fetch minibatch from sn data and train
        inputs, label_errors = sn_data.get_batch()
        feed_dict = self.safety_net.get_feed_dict(inputs, label_errors)
        _, l_sn = self.sess.run([self.safety_net.train_op, self.safety_net.loss_op], feed_dict=feed_dict)

        return l_clf

    def update_sn_params(self, inputs, label_errors):
        feed_dict = self.safety_net.get_feed_dict(inputs, label_errors)
        loss_value = self.safety_net.sess.run(self.safety_net.loss_op, feed_dict=feed_dict)

    def classify(self, inputs, features=False, reuse=False):
        with tf.variable_scope('classifier') as s:
            if reuse:
                s.reuse_variables()
            with tf.variable_scope('fc1') as scope:
                weights = weight_variable([self.input_dim, FC1])
                biases = bias_variable([FC1])
                pre_act_1 = tf.add(tf.matmul(inputs, weights), biases)
                fc1 = tf.nn.relu(pre_act_1, name=scope.name)
            with tf.variable_scope('fc2') as scope:
                weights = weight_variable([FC1, FC2])
                biases = bias_variable([FC2])
                pre_act_2 = tf.add(tf.matmul(fc1, weights), biases)
                fc2 = tf.nn.relu(pre_act_2, name=scope.name)
            with tf.variable_scope('fc3') as scope:
                weights = weight_variable([FC2, self.label_dim])
                biases = bias_variable([self.label_dim])
                pre_act_3 = tf.add(tf.matmul(fc2, weights), biases)
                fc3 = tf.identity(pre_act_3, name=scope.name)
        if features:
            return fc3, pre_act_2
        else: return fc3

    def compute_safety(self, inputs):
        _, features = self.classify(inputs, reuse=True, features=True)
        input_dim = features.get_shape().as_list()[1]
        with tf.variable_scope('safety_net'):
            with tf.variable_scope('fc1') as scope:
                weights = weight_variable([input_dim, FC1])
                biases = bias_variable([FC1])
                pre_act = tf.add(tf.matmul(features, weights), biases)
                fc1 = tf.nn.relu(pre_act, name=scope.name)
            with tf.variable_scope('fc2') as scope:
                weights = weight_variable([FC1, FC2])
                biases = bias_variable([FC2])
                pre_act = tf.add(tf.matmul(fc1, weights), biases)
                fc2 = tf.nn.relu(pre_act, name=scope.name)
            with tf.variable_scope('fc3') as scope:
                weights = weight_variable([FC2, 2])
                biases = bias_variable([2])
                pre_act = tf.add(tf.matmul(fc2, weights), biases)
                fc3 = tf.identity(pre_act, name=scope.name)
        return fc3

    def subset_selection(self, inputs, labels):
        feed_dict = {self.inputs: inputs, self.labels: labels}
        clf_errors = self.sess.run(self.clf_errors, feed_dict=feed_dict)
        subset_x = np.where(inputs, clf_errors==1)
        subset_y = np.where(labels, clf_errors==1)
        return subset_x, subset_y

    def add_to_classifier_data(self, inputs, labels):
        if self.classifier_data is None:
            self.classifier_data['x'] = inputs
            self.classifier_data['y'] = labels
        else:
            np.append(self.classifier_data['x'], inputs)
            np.append(self.classifier_data['y'], labels)

    def add_to_sn_data(self, inputs, labels):
        if self.sn_data is None:
            self.sn_data['x'] = inputs
            self.sn_data['y'] = labels
        else:
            np.append(self.sn_data['x'], inputs)
            np.append(self.sn_data['y'], labels)

    def get_clf_errors(self):
        with tf.variable_scope('inputs_to_query'):
            soft_errors = softmax_cross_entropy(logits=self.logits, labels=self.labels, reduce=False)
            hard_errors = tf.cast(soft_errors > self.error_threshold, tf.int32)
        return hard_errors

    def setup_evaluations(self):
        self.query_rate = fraction_above_zero(self.safety_net.logits)
        tf.add_to_collection('evaluation', self.query_rate)
        super(SafetyNetClassifier, self).setup_evaluations()


class Curriculum(Model):
    def __init__(self, input_shape, label_dim, student_lr, teacher_lr, train_teacher_every, n_batches=None):
        super(Curriculum, self).__init__(paradigm='classifier')

        # model hyperparameters
        self.teacher_temperature = 1
        self.conv = False
        self.self_study = True
        self.entropy_term = 0.0
        self.train_teacher_every = 10
        self.use_labels = False
        self.use_student_answers = False
        self.use_teacher_logits = True
        self.cheat = False
        assert (self.use_student_answers != self.use_teacher_logits)

        self.name = 'teacher_vs_student'
        self.inputs = tf.placeholder(tf.float32, [None] + input_shape)
        self.labels = tf.placeholder(tf.int32, [None, label_dim])
        self.input_shape = input_shape
        self.input_dim = np.prod(input_shape)
        self.teacher_input_dim = self.input_dim
        self.label_dim = label_dim
        self.n_batches = n_batches

        # optimization stuff
        self.student_lr = student_lr
        self.teacher_lr = teacher_lr
        if self.self_study: self.teacher_lr = 0.0
        self.student_optimizer = tf.train.AdamOptimizer(self.student_lr)
        self.teacher_optimizer = tf.train.AdamOptimizer(self.teacher_lr)

        if self.conv:
            self.student = self.conv_student
            self.teacher = self.conv_teacher
        else:
            self.student = self.fc_student
            self.teacher = self.fc_teacher

        with tf.variable_scope('student/') as self.student_scope:
            self.student_answers = self.student(self.inputs)

        with tf.variable_scope('teacher/') as self.teacher_scope:
            extra_info = []
            if self.use_labels:
                extra_info += [tf.cast(self.labels, tf.float32)]
                self.teacher_input_dim += self.label_dim
            if self.use_student_answers:
                extra_info += [self.student_answers]
                self.teacher_input_dim += self.label_dim

            self.teacher_logits = self.teacher(self.inputs, extra_info)

            teacher_weights = tf.nn.softmax(self.teacher_logits / self.teacher_temperature, dim=0)

            #self.normalization_constant = tf.Variable(0.0, name='normalization_constant')
            #self.sum_teacher_logits = tf.reduce_sum(self.teacher_logits)
            self.teacher_weights = tf.reshape(teacher_weights, [-1])

            self.weight_entropy = -tf.reduce_sum(self.teacher_weights * tf.log(self.teacher_weights))

        with tf.variable_scope('unweighted/') as self.unweighted_scope:
            self.unweighted_losses = softmax_cross_entropy(logits=self.student_answers, labels=self.labels, reduce=False)
            self.unweighted_loss = tf.reduce_mean(self.unweighted_losses)

        with tf.variable_scope('weighted/') as self.weighted_scope:
            self.weighted_losses = tf.mul(self.teacher_weights, self.unweighted_losses)
            self.weighted_loss = tf.reduce_sum(self.weighted_losses, axis=0)

        with tf.variable_scope(self.student_scope):
            if self.self_study:
                self.student_loss = self.unweighted_loss
            else: self.student_loss = self.weighted_loss
            self.loss_op = self.student_loss

        with tf.variable_scope(self.teacher_scope):
            self.teacher_loss = -self.weighted_loss - tf.mul(self.weight_entropy, self.entropy_term)

        teacher_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'teacher')
        student_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'student')
        self.train_student = self.train(self.student_loss, global_step=self.global_step, optimizer=self.student_optimizer, var_list=student_params)
        self.train_teacher = self.train(self.teacher_loss, global_step=self.global_step, optimizer=self.teacher_optimizer, var_list=teacher_params)

        #self.setup_evaluations()
        self.attach_summaries()
        self.summary_op = tf.summary.merge_all()
        self.check_op = tf.add_check_numerics_ops()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def update_params(self, inputs, labels):
        fd = self.get_feed_dict(inputs, labels)
        step = self.get_global_step()
        if step % self.train_teacher_every == 0:
            teacher_loss, _ = self.sess.run([self.teacher_loss, self.train_teacher], feed_dict=fd)
            student_loss, _ = self.sess.run([self.student_loss, self.train_student], feed_dict=fd)
        else:
            student_loss, _ = self.sess.run([self.student_loss, self.train_student], feed_dict=fd)
        return student_loss

    def get_feed_dict(self, inputs, labels):
        inputs = np.reshape(inputs, [-1] + self.input_shape)
        return {self.inputs: inputs, self.labels: labels}

    def fc_teacher(self, inputs, extra_info=None):
        #if reuse:
        #    s.reuse_variables()
        reshape = tf.reshape(inputs, [-1, self.input_dim])
        if extra_info is not None:
            reshape = tf.concat(concat_dim=1, values=[reshape, *extra_info])
        with tf.variable_scope('fc1') as scope:
            weights = weight_variable([self.teacher_input_dim, FC1])
            biases = bias_variable([FC1])
            pre_act = tf.add(tf.matmul(reshape, weights), biases)
            fc1 = tf.nn.relu(pre_act, name=scope.name)
        with tf.variable_scope('fc2') as scope:
            weights = weight_variable([FC1, FC2])
            biases = bias_variable([FC2])
            pre_act = tf.add(tf.matmul(fc1, weights), biases)
            fc2 = tf.nn.relu(pre_act, name=scope.name)
        with tf.variable_scope('fc3') as scope:
            weights = weight_variable([FC2, 1])
            biases = bias_variable([1])
            pre_act = tf.add(tf.matmul(fc2, weights), biases)
            fc3 = tf.identity(pre_act, name=scope.name)
        return fc3

    def conv_teacher(self, inputs, extra_info=None):
        reshape = tf.reshape(inputs, [-1] + self.input_shape)
        with tf.variable_scope('conv1') as scope:
            kernel = weight_variable(CONV1)
            conv = tf.nn.conv2d(reshape, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable(CONV1[3])
            pre_act = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_act, name=scope.name)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

        with tf.variable_scope('conv2') as scope:
            kernel = weight_variable(CONV2)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable(CONV2[3])
            pre_act = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_act, name=scope.name)

        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        with tf.variable_scope('fc3') as scope:
            dim = np.prod(pool2.get_shape()[1:].as_list())
            reshape = tf.reshape(pool2, [-1, dim])
            weights = weight_variable([dim, 1])
            biases = bias_variable([1])
            pre_act = tf.add(tf.matmul(reshape, weights), biases)
            fc3 = tf.identity(pre_act, name=scope.name)

        return fc3

    def fc_student(self, inputs, extra_info=None, dropout=False):

        #if reuse:
        #    s.reuse_variables()
        reshape = tf.reshape(inputs, [-1, self.input_dim])
        if extra_info is not None:
            reshape = tf.concat(concat_dim=1, values=[reshape, *extra_info])
        input_dim = reshape.get_shape().as_list()[1]
        with tf.variable_scope('fc1') as scope:
            weights = weight_variable([input_dim, FC1])
            biases = bias_variable([FC1])
            pre_act = tf.add(tf.matmul(reshape, weights), biases)
            fc1 = tf.nn.relu(pre_act, name=scope.name)
            if dropout:
                fc1 = tf.nn.dropout(fc1, keep_prob=0.5)
        with tf.variable_scope('fc2') as scope:
            weights = weight_variable([FC1, FC2])
            biases = bias_variable([FC2])
            pre_act = tf.add(tf.matmul(fc1, weights), biases)
            fc2 = tf.nn.relu(pre_act, name=scope.name)
            if dropout:
                fc2 = tf.nn.dropout(fc2, keep_prob=0.5)
        with tf.variable_scope('fc3') as scope:
            weights = weight_variable([FC2, self.label_dim])
            biases = bias_variable([self.label_dim])
            pre_act = tf.add(tf.matmul(fc2, weights), biases)
            fc3 = tf.identity(pre_act, name=scope.name)
        return fc3

    def conv_student(self, inputs, extra_info=None):
        reshape = tf.reshape(inputs, [-1] + self.input_shape)
        with tf.variable_scope('conv1') as scope:
            kernel = weight_variable(CONV1)
            conv = tf.nn.conv2d(reshape, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable(CONV1[3])
            pre_act = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_act, name=scope.name)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

        with tf.variable_scope('conv2') as scope:
            kernel = weight_variable(CONV2)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = bias_variable(CONV2[3])
            pre_act = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_act, name=scope.name)

        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        with tf.variable_scope('fc3') as scope:
            dim = np.prod(pool2.get_shape()[1:].as_list())
            reshape = tf.reshape(pool2, [-1, dim])
            weights = weight_variable([dim, self.label_dim])
            biases = bias_variable([self.label_dim])
            pre_act = tf.add(tf.matmul(reshape, weights), biases)
            fc3 = tf.identity(pre_act, name=scope.name)

        return fc3

    def evaluate(self, inputs, labels):
        feed_dict=self.get_feed_dict(inputs, labels)
        loss_value, summary_str = self.sess.run([self.loss_op, self.summary_op], feed_dict=feed_dict)
        return loss_value, summary_str

    def setup_evaluations(self):
        evaluation_list = tf.get_collection('evaluation')
        with tf.variable_scope('unweighted', reuse=True):
            self.accuracy = accuracy_with_logits(logits=self.student_answers, labels=self.labels)
            tf.summary.scalar('accuracy', self.accuracy)
        tf.add_to_collection('evaluation', self.accuracy)
        with tf.variable_scope('weighted', reuse=True):
            self.weighted_accuracy = accuracy_with_logits(logits=self.student_answers, labels=self.labels, weights=self.teacher_weights)
            tf.summary.scalar('accuracy', self.weighted_accuracy)
        tf.add_to_collection('evaluation', self.weighted_accuracy)
        evaluation_list = tf.get_collection('evaluation')
        self.evaluation_dict = {evaluation.name: evaluation for evaluation in evaluation_list}
        assert len(self.evaluation_dict.keys()) > 0
        self.evaluations_have_been_setup = True

    def model_description(self):
        settings = {}
        settings['input_shape'] = self.input_shape
        settings['label_dim'] = self.label_dim
        settings['train_teacher_every'] = self.train_teacher_every
        settings['student_lr'] = self.student_lr
        settings['teacher_lr'] = self.teacher_lr
        settings['student_optimizer'] = 'adam'
        settings['teacher_optimizer'] = 'adam'
        settings['entropy_term'] = self.entropy_term
        settings['conv'] = self.conv
        settings['name'] = self.name
        settings['teacher_temperature'] = self.teacher_temperature
        settings['use_labels'] = self.use_labels
        settings['use_student_answers'] = self.use_student_answers
        settings['cheat'] = self.cheat
        return settings

    def attach_summaries(self):
        with tf.variable_scope(self.unweighted_scope):
            tf.summary.histogram('cross_entropy_batched', self.unweighted_losses)
            max_l = tf.reduce_max(self.unweighted_losses)
            min_l = tf.reduce_min(self.unweighted_losses)
            mean, variance = tf.nn.moments(self.unweighted_losses, axes=[0])
            tf.summary.scalar('max_loss', max_l)
            tf.summary.scalar('min_loss', min_l)
            tf.summary.scalar('loss_variance', variance)
            tf.summary.scalar('cross_entropy', self.unweighted_loss)
            self.accuracy = accuracy_with_logits(logits=self.student_answers, labels=self.labels)
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.variable_scope(self.weighted_scope):
            max_l = tf.reduce_max(self.weighted_losses)
            min_l = tf.reduce_min(self.weighted_losses)
            mean, variance = tf.nn.moments(self.weighted_losses, axes=[0])
            tf.summary.scalar('max_loss', max_l)
            tf.summary.scalar('min_loss', min_l)
            tf.summary.scalar('loss_variance', variance)
            tf.summary.histogram('cross_entropy_batched', self.weighted_losses)
            tf.summary.scalar('cross_entropy', self.weighted_loss)
            self.weighted_accuracy = accuracy_with_logits(logits=self.student_answers, labels=self.labels, weights=self.teacher_weights)
            tf.summary.scalar('accuracy', self.weighted_accuracy)

        with tf.variable_scope(self.student_scope):
            tf.summary.scalar('loss', self.student_loss)

        with tf.variable_scope(self.teacher_scope):
            tf.summary.histogram('teacher_weights', self.teacher_weights)
            tf.summary.scalar('max_weight', tf.reduce_max(self.teacher_weights))
            tf.summary.scalar('min_weight', tf.reduce_min(self.teacher_weights))
            self.tw_mean, self.tw_variance = tf.nn.moments(self.teacher_weights, axes=[0])
            tf.summary.scalar('mean_weight', self.tw_mean)
            tf.summary.scalar('weight_variance', self.tw_variance)
            tf.summary.scalar('weight_entropy', self.weight_entropy)
            tf.summary.histogram('teacher_logits', self.teacher_logits)
            tf.summary.scalar('loss', self.teacher_loss)





class Generative(Model):
    def __init__(self, input_dim, hidden_dim):
        super(Generative, self).__init__(paradigm='generative')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.x = tf.placeholder(tf.float32, [None, input_dim], 'x')

    def generate_and_save_inputs(self, num_samples, directory):
        '''Generates the inputs using the model and saves them in the directory
        Args:
            num_samples: number of samples to generate
            directory: a directory to save the inputs
        '''
        try: generator = self.fake # GAN, VAE
        except AttributeError: generator = None
        if generator is None: raise AttributeError

        imgs = self.sess.run(generator, feed_dict={self.noise: np.random.rand(num_samples, self.hidden_dim)})
        save_visualizations(imgs, directory)

    def get_feed_dict(self, inputs, *args):
        noise = np.random.rand(inputs.shape[0], self.hidden_dim)
        feed_dict={self.x: inputs, self.noise: noise}
        return feed_dict

    def evaluate(self, inputs, *args):
        feed_dict=self.get_feed_dict(inputs)
        evaluations = {evaluation_name: self.sess.run(evaluation, feed_dict) for evaluation_name, evaluation in self.evaluation_dict.items()}
        return evaluations

class GAN(Generative):
    def __init__(self, input_dim, hidden_dim, G_opt, D_opt):
        super(GAN, self).__init__(input_dim, hidden_dim)

        self.G_optimizer = G_opt
        self.D_optimizer = D_opt

        self.real = tf.identity(self.x, name='real')
        self.noise = tf.placeholder(tf.float32, (None, hidden_dim), name='noise')

        self.D_real = self.discriminator(self.real, reuse=False)
        self.fake = self.generator(self.noise)
        self.D_fake = self.discriminator(self.fake, reuse=True)

        self.G_loss = generator_loss(self.fake)
        self.D_loss = discriminator_loss(self.D_real, self.D_fake)

        self.D_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'discriminator')
        self.G_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'generator')

        self.G_train = self.train(self.G_loss, global_step=self.global_step, optimizer=self.G_optimizer, var_list=self.G_params)
        self.D_train = self.train(self.D_loss, global_step=self.global_step, optimizer=self.D_optimizer, var_list=self.D_params)

        self.setup_evaluations()
        self.setup_summaries()

        self.summary_op = tf.summary.merge_all()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def discriminator(self, inputs, reuse=False):
        with tf.variable_scope('discriminator') as s:
            if reuse:
                s.reuse_variables()
            with tf.variable_scope('fc1') as scope:
                weights = weight_variable([28 * 28, FC1])
                biases = bias_variable([FC1])
                pre_act = tf.add(tf.matmul(inputs, weights), biases)
                fc1 = tf.nn.relu(pre_act, name=scope.name)
            with tf.variable_scope('fc2') as scope:
                weights = weight_variable([FC1, FC2])
                biases = bias_variable([FC2])
                pre_act = tf.add(tf.matmul(fc1, weights), biases)
                fc2 = tf.nn.relu(pre_act, name=scope.name)
            with tf.variable_scope('fc3') as scope:
                weights = weight_variable([FC2, 1])
                biases = bias_variable([1])
                pre_act = tf.add(tf.matmul(fc2, weights), biases)
                fc3 = tf.identity(pre_act, name=scope.name)
        return fc3

    def generator(self, noise):
        with tf.variable_scope('generator'):
            with tf.variable_scope('fc1') as scope:
                weights = weight_variable([HIDDEN_SIZE, FC1])
                biases = bias_variable([FC1])
                pre_act = tf.add(tf.matmul(noise, weights), biases)
                fc1 = tf.nn.relu(pre_act, name=scope.name)
            with tf.variable_scope('fc2') as scope:
                weights = weight_variable([FC1, FC2])
                biases = bias_variable([FC2])
                pre_act = tf.add(tf.matmul(fc1, weights), biases)
                fc2 = tf.nn.relu(pre_act, name=scope.name)
            with tf.variable_scope('fc3') as scope:
                weights = weight_variable([FC2, IMAGE_SIZE])
                biases = bias_variable([IMAGE_SIZE])
                pre_act = tf.add(tf.matmul(fc2, weights), biases)
                fc3 = tf.identity(pre_act, name=scope.name)
        return fc3

    def update_params(self, inputs, *args):
        feed_dict = self.get_feed_dict(inputs)
        d_loss_value, _ = self.sess.run([self.D_loss, self.D_train], feed_dict)
        g_loss_value, _ = self.sess.run([self.G_loss, self.G_train], feed_dict)
        return g_loss_value

    def setup_summaries(self):
        assert self.evaluations_have_been_setup
        tf.summary.scalar('D_loss', self.D_loss)
        tf.summary.scalar('G_loss', self.G_loss)
        tf.summary.scalar('FP', self.fp)
        tf.summary.scalar('TN', self.tn)
        tf.summary.scalar('F0', self.f0)
        tf.summary.scalar('TP', self.tp)
        tf.summary.scalar('FN', self.fn)
        tf.summary.scalar('T0', self.t0)

    def setup_evaluations(self):
        evaluation_list = tf.get_collection('evaluation')
        tf.add_to_collection('evaluation', self.D_loss)
        tf.add_to_collection('evaluation', self.G_loss)
        self.fp = tf.size(tf.where(self.D_fake > 0), name='FP')
        self.f0 = tf.size(tf.where(tf.equal(self.D_fake, 0)), name='fake_zeros')
        self.tn = tf.size(tf.where(self.D_fake < 0), name='TN')
        #f = tf.size(self.D_fake, name='F')
        tf.add_to_collection('evaluation', self.fp)
        tf.add_to_collection('evaluation', self.tn)
        tf.add_to_collection('evaluation', self.f0)

        self.tp = tf.size(tf.where(self.D_real > 0), name='TP')
        self.t0 = tf.size(tf.where(tf.equal(self.D_real, 0)), name='real_zeros')
        self.fn = tf.size(tf.where(self.D_real < 0), name='FN')
        #f = tf.size(self.D_fake, name='F')
        tf.add_to_collection('evaluation', self.fn)
        tf.add_to_collection('evaluation', self.tp)
        tf.add_to_collection('evaluation', self.t0)
        #tf.add_to_collection('evaluation', f)
        evaluation_list = tf.get_collection('evaluation')
        self.evaluation_dict = {evaluation.name: evaluation for evaluation in evaluation_list}
        assert len(self.evaluation_dict.keys()) > 0

        self.evaluations_have_been_setup = True

class VAE(Generative):
    def __init__(self, input_dim, hidden_dim, optimizer):
        super(VAE, self).__init__(input_dim, hidden_dim)
        self.optimizer = optimizer

        #self.x = tf.placeholder(tf.float32, [None, input_dim], 'x')
        self.noise = tf.placeholder(tf.float32, (None, hidden_dim), 'noise')

        self.z_mean, self.z_log_sigma_sq = tf.split(split_dim=1, num_split=2, value=self.recognition_network(self.x))
        self.z = tf.add(self.z_mean,
                    tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), self.noise))
        self.x_rec = self.generator_network(self.z)
        self.fake = self.generator_network(self.noise, reuse=True)

        self.latent_loss = latent_loss(self.z_mean, self.z_log_sigma_sq, reduce=True)
        self.reconstruction_loss = reconstruction_loss(self.x, self.x_rec, reduce=True)

        self.loss_op = tf.reduce_mean(self.latent_loss + self.reconstruction_loss, name='total_loss')
        self.train_op = self.train(self.loss_op, global_step=self.global_step, optimizer=self.optimizer)

        self.setup_evaluations()
        self.setup_summaries()

        self.summary_op = tf.summary.merge_all()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def recognition_network(self, inputs, reuse=False):
        with tf.variable_scope('inference') as s:
            if reuse:
                s.reuse_variables()
            with tf.variable_scope('fc1') as scope:
                weights = weight_variable([IMAGE_SIZE, FC1])
                biases = bias_variable([FC1])
                pre_act = tf.add(tf.matmul(inputs, weights), biases)
                fc1 = tf.nn.relu(pre_act, name=scope.name)
            with tf.variable_scope('fc2') as scope:
                weights = weight_variable([FC1, FC2])
                biases = bias_variable([FC2])
                pre_act = tf.add(tf.matmul(fc1, weights), biases)
                fc2 = tf.nn.relu(pre_act, name=scope.name)
            with tf.variable_scope('fc3') as scope:
                weights = weight_variable([FC2, HIDDEN_SIZE * 2])
                biases = bias_variable([HIDDEN_SIZE * 2])
                pre_act = tf.add(tf.matmul(fc2, weights), biases)
                fc3 = tf.identity(pre_act, name=scope.name)
        return fc3

    def generator_network(self, hidden, reuse=False):
        with tf.variable_scope('generation') as s:
            if reuse:
                s.reuse_variables()
            with tf.variable_scope('fc1') as scope:
                weights = weight_variable([HIDDEN_SIZE, FC1])
                biases = bias_variable([FC1])
                pre_act = tf.add(tf.matmul(hidden, weights), biases)
                fc1 = tf.nn.relu(pre_act, name=scope.name)
            with tf.variable_scope('fc2') as scope:
                weights = weight_variable([FC1, FC2])
                biases = bias_variable([FC2])
                pre_act = tf.add(tf.matmul(fc1, weights), biases)
                fc2 = tf.nn.relu(pre_act, name=scope.name)
            with tf.variable_scope('fc3') as scope:
                weights = weight_variable([FC2, IMAGE_SIZE])
                biases = bias_variable([IMAGE_SIZE])
                pre_act = tf.add(tf.matmul(fc2, weights), biases)
                fc3 = tf.nn.sigmoid(pre_act, name=scope.name)
        return fc3

    def update_params(self, inputs, *args):
        feed_dict = self.get_feed_dict(inputs)
        loss_value, _ = self.sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss_value

    def setup_evaluations(self):
        evaluation_list = tf.get_collection('evaluation')
        tf.add_to_collection('evaluation', self.latent_loss)
        tf.add_to_collection('evaluation', self.reconstruction_loss)
        tf.add_to_collection('evaluation', self.loss_op)
        evaluation_list = tf.get_collection('evaluation')
        self.evaluation_dict = {evaluation.name: evaluation for evaluation in evaluation_list}
        assert len(self.evaluation_dict.keys()) > 0
        self.evaluations_have_been_setup = True

    def setup_summaries(self):
        tf.summary.scalar('latent_loss', self.latent_loss)
        tf.summary.scalar('reconstruction_loss', self.reconstruction_loss)
        tf.summary.scalar('total_loss', self.loss_op)
