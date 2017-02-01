import numpy as np
import tensorflow as tf
import time
import datetime
import inspect
from tensorflow.examples.tutorials.mnist import input_data
import os
from loss import *
from models import *
from constants import *
from datasets import mnist
from utils import save_visualizations

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 20, "max epoch")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_string("model", "", "")
flags.DEFINE_string("tt", 10, "train teacher every")

FLAGS = flags.FLAGS
DATA_DIR = os.path.join(FLAGS.working_directory, "MNIST")
RES_DIR = os.path.join(FLAGS.working_directory, "new_results")
MNIST = input_data.read_data_sets(DATA_DIR, one_hot=True)

class Experiment:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.max_epoch = FLAGS.max_epoch
        self.updates_per_epoch = FLAGS.updates_per_epoch
        self.train_eval_freq = EVAL_FREQ
        self.evals_per_epoch = EVAL_BATCHES
        self.batch_size = FLAGS.batch_size
        self.do_admin()

    def write_settings(self):
        settings = self.model.model_description()
        with open(os.path.join(RES_DIR, self.unique_id, 'settings.txt'), 'w') as f:
            for name in sorted(settings.keys()):
                f.write("{}: {}\n".format(name, settings[name]))

    def do_admin(self):
        if not os.path.exists(RES_DIR):
            os.makedirs(RES_DIR)
        dt = datetime.datetime.now()
        self.unique_id = '{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)
        self.train_writer = tf.summary.FileWriter(os.path.join(RES_DIR, self.unique_id, 'train'), self.model.sess.graph)
        self.valid_writer = tf.summary.FileWriter(os.path.join(RES_DIR, self.unique_id, 'valid'), self.model.sess.graph)
        self.write_settings()

    def train(self, epoch):
        step = epoch * self.updates_per_epoch
        stride = self.train_eval_freq
        training_loss = 0.0
        for i in range(self.updates_per_epoch):
            images, labels = self.data.train.next_batch(self.batch_size)
            loss_value = self.model.update_params(images, labels)
            training_loss += loss_value

            if i % (self.train_eval_freq) == 0: # evaluation
                _, t_summary_str = self.model.evaluate(images, labels)
                self.train_writer.add_summary(t_summary_str, global_step=step)
                step += stride

        training_loss /= self.updates_per_epoch
        print("Training loss: {}".format(training_loss))

    def validate(self, epoch):
        step = epoch * self.updates_per_epoch
        stride = self.updates_per_epoch / self.evals_per_epoch
        validation_loss = 0.0
        for i in range(self.evals_per_epoch):
            images, labels = self.data.test.next_batch(self.batch_size)
            loss_value, summary_str = self.model.evaluate(images, labels)
            self.valid_writer.add_summary(summary_str, global_step=step)
            validation_loss += loss_value
            step += stride
        validation_loss /= (self.evals_per_epoch)
        print("Validation loss: {}".format(validation_loss))

    def run(self):
        for epoch in range(self.max_epoch):
            print('\nEpoch {}'.format(epoch))
            t = time.time()
            self.train(epoch)
            self.validate(epoch)
            t = time.time() - t

            print("Processed {} examples per second. Epoch took {} seconds".format(
                    (self.updates_per_epoch + self.evals_per_epoch) * self.batch_size / t, t))
            if self.model.paradigm == 'generative':
                self.model.generate_and_save_images(
                    FLAGS.batch_size, FLAGS.working_directory)

def get_model(model_string):
    opt1 = tf.train.AdamOptimizer(FLAGS.learning_rate)
    opt2 = tf.train.AdamOptimizer(FLAGS.learning_rate*0.0)
    lr1 = FLAGS.learning_rate
    lr2 = FLAGS.learning_rate
    if model_string == "gan":
        model = GAN(IMAGE_SIZE, HIDDEN_SIZE, opt1, opt2)
    elif model_string == "vae":
        model = VAE(IMAGE_SIZE, HIDDEN_SIZE, opt1)
    elif model_string == "fc":
        model = FullyConnectedClassifier(input_shape=IMAGE_SHAPE,
                                        label_dim=NUM_CLASSES,
                                        optimizer=opt1)
    elif model_string == "conv":
        model = ConvnetClassifier(input_shape=IMAGE_SHAPE, label_dim=NUM_CLASSES, optimizer=opt1)
    elif model_string == 'safe':
        model = SafetyNetClassifier(cost_per_query=0.3,
                                    input_dim=IMAGE_SIZE,
                                    label_dim=NUM_CLASSES,
                                    optimizer=opt1,
                                    safety_net_optimizer=opt2)
    elif model_string == 'cu':
        model = Curriculum(input_shape=IMAGE_SHAPE,
                           label_dim=NUM_CLASSES,
                           student_lr=lr1,
                           teacher_lr=lr2,
                           train_teacher_every=FLAGS.tt)

    else: raise NameError

    return model


def main():
    data = MNIST
    #data = mnist(FLAGS.batch_size, data_directory)
    model = get_model(FLAGS.model)
    experiment = Experiment(model, data)
    experiment.run()

if __name__ == '__main__':
    main()
