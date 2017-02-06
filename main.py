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
from datasets import *
from utils import save_visualizations

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 50, "max epoch")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_string("model", "", "")
flags.DEFINE_integer("tte", 10, "train teacher every")

FLAGS = flags.FLAGS
RES_DIR = os.path.join(FLAGS.working_directory, "new_results")
#MNIST_ = input_data.read_data_sets(DATA_DIR, one_hot=True)

class Experiment:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        #self.training_data, self.test_data = data
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
            f.write("Dataset: {}".format(self.data.name))

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
        total_confusion = np.zeros((self.model.label_dim, self.model.label_dim))
        for i in range(self.updates_per_epoch):
            images, labels = self.data.get_batch()
            try: loss_value = self.model.update_params(images, labels)
            except Exception as e:
                print(e)
                import ipdb; ipdb.set_trace()
            training_loss += loss_value

            if i % (self.train_eval_freq) == 0: # evaluation
                _, t_summary_str, confusion = self.model.evaluate(images, labels)
                self.train_writer.add_summary(t_summary_str, global_step=step)
                step += stride
                total_confusion += confusion

        training_loss /= self.updates_per_epoch
        print("Training loss: {:.5f}".format(training_loss))
        print("Confusion Matrix: {} examples total\n {}".format(int(np.sum(total_confusion)), total_confusion.astype('int32')))

    def validate(self, epoch):
        step = epoch * self.updates_per_epoch
        stride = self.updates_per_epoch / self.evals_per_epoch
        validation_loss = 0.0
        total_confusion = np.zeros((self.model.label_dim, self.model.label_dim))
        for i in range(self.evals_per_epoch):
            images, labels = self.data.get_batch(test=True)
            loss_value, summary_str, confusion = self.model.evaluate(images, labels)
            self.valid_writer.add_summary(summary_str, global_step=step)
            validation_loss += loss_value
            step += stride
            total_confusion += confusion
        validation_loss /= (self.evals_per_epoch)
        print("Validation loss: {:.5f}".format(validation_loss))
        print("Confusion Matrix: {} examples total\n {}".format(int(np.sum(total_confusion)), total_confusion.astype('int32')))

    def run(self):
        for epoch in range(self.max_epoch):
            print('\nEpoch {}'.format(epoch))
            t = time.time()
            self.train(epoch)
            self.validate(epoch)
            t = time.time() - t
            print("Processed {:.0f} examples per second. Epoch took {:.2f} seconds".format(
                    (self.updates_per_epoch + self.evals_per_epoch) * self.batch_size / t, t))
            if self.model.paradigm == 'generative':
                self.model.generate_and_save_images(
                    self.batch_size, FLAGS.working_directory)

def main():
    #data = MNIST(FLAGS.batch_size, DATA_DIR, dims=2, one_hot=True, unbalanced_train=0.5, unbalanced_test=0.5)
    #data = CIFAR10(FLAGS.batch_size, DATA_DIR, one_hot=True)
    data = CIFAR10_u05(FLAGS.batch_size, DATA_DIR, one_hot=True)
    model = Curriculum(input_shape=data.input_shape,
                           label_dim=data.label_dim,
                           student_lr=FLAGS.learning_rate,
                           teacher_lr=FLAGS.learning_rate,
                           train_teacher_every=FLAGS.tte,
                           conv=False,
                           self_study=False,
                           teacher_temperature=1.0,
                           entropy_term=0.1,
                           use_labels=False,
                           use_student_answers=False,
                           l1_reg=0.0)
    experiment = Experiment(model, data)
    experiment.run()

if __name__ == '__main__':
    main()
