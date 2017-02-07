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

flags.DEFINE_boolean("debug", False, "debug mode")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 50, "max epoch")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_string("model", "", "")
flags.DEFINE_integer("tte", 10, "train teacher every")
flags.DEFINE_boolean("conv", False, "convolutional")
flags.DEFINE_boolean("ss", False, "self study")
flags.DEFINE_float("ent", 0.0, "entropy term")
flags.DEFINE_float("tt", 1.0, "teacher temperature")
flags.DEFINE_boolean('stu', False, "use student answers")
flags.DEFINE_boolean("lab", False, "use labels")
flags.DEFINE_boolean("mnist", False, "mnist")
flags.DEFINE_boolean("cifar", False, "cifar")
flags.DEFINE_integer("gpu", 0, "which gpu")
flags.DEFINE_boolean("restore", False, "restore")


FLAGS = flags.FLAGS

if FLAGS.debug:
    tf.logging.set_verbosity(tf.logging.INFO)
RES_DIR = os.path.join(FLAGS.working_directory, "new_results")
DEVICE = '/gpu:{}'.format(FLAGS.gpu)
if FLAGS.gpu == -1: DEVICE = '/cpu:0'
#MNIST_ = input_data.read_data_sets(DATA_DIR, one_hot=True)

class Experiment:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        #self.training_data, self.test_data = data
        self.max_epoch = FLAGS.max_epoch
        self.save_every = 1
        self.updates_per_epoch = FLAGS.updates_per_epoch
        self.train_eval_freq = EVAL_FREQ
        self.evals_per_epoch = EVAL_BATCHES
        self.batch_size = FLAGS.batch_size
        self.saver = tf.train.Saver()
        self.do_admin()

    def write_settings(self):
        settings = self.model.model_description()
        with open(os.path.join(self.experiment_dir, 'settings.txt'), 'w') as f:
            for name in sorted(settings.keys()):
                f.write("{}: {}\n".format(name, settings[name]))
            f.write("Batch size: {}".format(self.batch_size))
            f.write("Updates per epoch: {}".format(self.updates_per_epoch))
            f.write("Training evaluation frequency: {}".format(self.train_eval_freq))
            f.write("Max epochs: {}".format(self.max_epoch))
            f.write("Dataset: {}".format(self.data.name))
            f.write("Device: {}".format(DEVICE))


    def do_admin(self):
        if not os.path.exists(RES_DIR):
            os.makedirs(RES_DIR)
        dt = datetime.datetime.now()
        self.unique_id = '{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)
        self.experiment_dir = os.path.join(RES_DIR, self.unique_id)
        self.train_writer = tf.summary.FileWriter(os.path.join(self.experiment_dir, 'train'), self.model.sess.graph)
        self.valid_writer = tf.summary.FileWriter(os.path.join(self.experiment_dir, 'valid'), self.model.sess.graph)
        self.write_settings()

    def train(self, epoch):
        step = epoch * self.updates_per_epoch
        stride = self.train_eval_freq
        training_loss = 0.0
        total_confusion = np.zeros((self.model.label_dim, self.model.label_dim))
        for i in range(self.updates_per_epoch):
            images, labels = self.data.get_batch()
            #import ipdb; ipdb.set_trace()
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
        if FLAGS.restore:
            latest_checkpoint = tf.train.latest_checkpoint(self.experiment_dir)
            self.saver.restore(self.model.sess, latest_checkpoint)
            print("Restored model from checkpoint {}".format(latest_checkpoint))

        for epoch in range(self.max_epoch):
            print('\nEpoch {}'.format(epoch))
            t = time.time()
            self.train(epoch)
            self.validate(epoch)
            t = time.time() - t
            print("Processed {:.0f} examples per second. Epoch took {:.2f} seconds".format(
                    (self.updates_per_epoch + self.evals_per_epoch) * self.batch_size / t, t))
            if epoch % self.save_every == 0:
                self.saver.save(self.model.sess, self.experiment_dir + '/model', global_step=epoch)

def main():
    sess = tf.Session(config=tf.ConfigProto(
                        log_device_placement=FLAGS.debug,
                        allow_soft_placement=True))
    with tf.device(DEVICE):
        if FLAGS.mnist:
            data = MNIST(FLAGS.batch_size, DATA_DIR, dims=2, one_hot=True, unbalanced_train=0.5, unbalanced_test=0.5)
        elif FLAGS.cifar:
            data = CIFAR10(FLAGS.batch_size, DATA_DIR, one_hot=True)
            #data = CIFAR10_u05(FLAGS.batch_size, DATA_DIR, one_hot=True)
        else: raise ValueError("You must select a dataset!")
        if FLAGS.debug: data.unit_test(True)
        model = Curriculum(sess=sess,
                            input_shape=data.input_shape,
                               label_dim=data.label_dim,
                               student_lr=FLAGS.learning_rate,
                               teacher_lr=FLAGS.learning_rate,
                               train_teacher_every=FLAGS.tte,
                               conv=FLAGS.conv,
                               self_study=FLAGS.ss,
                               teacher_temperature=FLAGS.tt,
                               entropy_term=FLAGS.ent,
                               use_labels=FLAGS.lab,
                               use_student_answers=FLAGS.stu,
                               l1_reg=0.0)
        experiment = Experiment(model, data)
        experiment.run()

if __name__ == '__main__':
    main()
