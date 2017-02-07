from collections import namedtuple
from utils import whiten
from struct import unpack
import gzip
from numpy import zeros, uint8, float32
import os
import numpy as np
#from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
#from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = os.path.join('', "data")
DATASET_TYPES = ['train', 'valid', 'test']
class UnlabelledDataSet(object):

  def __init__(self,
               images):
    self._num_examples = images.shape[0]
    self._images = images
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end]


class ProtoLabelledDataset(object):
    def __init__(self, x=None, y=None, batch_size=1):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.total_data_seen = 0
        self.reset_counters()
        if x is not None and y is not None: self.reset_size_info()

    def add_data(self, x, y):
        assert x.shape[0] == y.shape[0]

        if self.x is None:
            self.x = x
        else: np.append(self.x, x)

        if self.y is None:
            self.y = y
        else: np.append(self.y, y)

        self.reset_size_info()

    def reset_size_info(self):
        self.n_data = self.x.shape[0]
        self.n_batches = self.n_data / self.batch_size

    def reset_counters(self):
        self.on_batch = 0
        self.on_index = 0

    def get_batch(self, random=False):
        if self.on_batch > self.n_batches:
            self.reset_counters()

        if not random:
            start, end = self.on_index, self.on_index + self.batch_size
            x, y = self.x[start:end], self.y[start:end]
        else:
            perm = np.random.choice(self.n_data, self.batch_size)
            x, y = self.x[perm], self.y[perm]

        self.on_batch += 1
        self.on_index += self.batch_size
        self.total_data_seen += self.batch_size

        return x, y

class LabelledDataset(object):
    def __init__(self, batch_size, data_dir, name, dims, input_shape, label_dim,
                one_hot=True):
        self.data_dir = data_dir
        self.name = name
        self.dims = dims
        self.one_hot = one_hot

        self.load_train_data(data_dir)
        self.load_test_data(data_dir)

        self.input_shape = input_shape
        self.input_dim = np.prod(self.input_shape)
        self.label_dim = label_dim

        self.n_train = self.x_train.shape[0]
        self.n_test = self.x_test.shape[0]

        self.reset_counters()
        self.batch_size = batch_size
        self.n_train_batches = int(self.n_train / batch_size)
        self.n_test_batches = int(self.n_test / batch_size)

    def load_train_data(self, data_dir):
        self.x_train = np.load(os.path.join(data_dir, 'x-train.npy'))
        self.y_train = np.load(os.path.join(data_dir, 'y-train.npy'))

    def load_test_data(self, data_dir):
        self.x_test = np.load(os.path.join(data_dir, 'x-test.npy'))
        self.y_test = np.load(os.path.join(data_dir, 'y-test.npy'))

    def x_preprocess(self, x):
        if self.dims == 2:
            return np.reshape(x, [x.shape[0], -1])
        elif self.dims == 3:
            return np.reshape(x, [x.shape[0], x.shape[1], -1])
        elif self.dims == 4:
            return np.reshape(x, [x.shape[0], x.shape[1], x.shape[2], -1])
        else: raise ValueError

    @staticmethod
    def dense_to_one_hot(labels_dense, num_classes):
          """Convert class labels from scalars to one-hot vectors."""
          num_labels = labels_dense.shape[0]
          index_offset = np.arange(num_labels) * num_classes
          labels_one_hot = np.zeros((num_labels, num_classes))
          labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
          return labels_one_hot

    def y_preprocess(self, y):
        if self.one_hot:
            return self.dense_to_one_hot(y, self.label_dim)
        else: return y

    def reset_counters(self):
        self.on_train_batch = 0
        self.on_test_batch = 0
        self.on_epoch = 0
        self.on_train_index = 0
        self.on_test_index = 0

    def set_batch_size(self, batch_size):
        self.reset_counters()
        self.batch_size = batch_size
        self.n_train_batches = int(self.x_train.shape[0] / batch_size)
        self.n_test_batches = int(self.x_test.shape[0] / batch_size)

    def get_batch(self, test=False, random=False):
        if test:
            return self.get_test_batch(random)
        else:
            return self.get_train_batch(random)

    def get_train_batch(self, random=False):
        start, end = self.on_train_index, self.on_train_index + self.batch_size
        self.on_train_batch += 1
        self.on_train_index += self.batch_size
        if self.on_train_batch >= self.n_train_batches:
            self.on_train_index = 0
            self.on_train_batch = 0
            self.on_epoch += 1

        return self.x_preprocess(self.x_train[start:end]), self.y_preprocess(self.y_train[start:end])

    def get_test_batch(self, random=False):
        start, end = self.on_test_index, self.on_test_index + self.batch_size
        self.on_test_batch += 1
        self.on_test_index += self.batch_size
        if self.on_test_batch >= self.n_test_batches:
            self.on_test_index = 0
            self.on_test_batch = 0
            self.on_epoch += 1

        return self.x_preprocess(self.x_test[start:end]), self.y_preprocess(self.y_test[start:end])

    def unit_test(self, vis=False, updates_per_epoch=100, epochs=10):

        try:
            assert self.on_train_batch == 0
            for i in range(epochs):
                #assert self.on_epoch == i
                for j in range(updates_per_epoch):
                    got_batch = False
                    k = i * updates_per_epoch
                    assert self.on_train_batch == (j + k) % self.n_train_batches
                    assert self.on_train_index == ((j + k) * self.batch_size) % self.n_train
                    images, labels = self.get_batch()
                    got_batch = True
                    assert self.on_train_batch == (j + 1 + k) % self.n_train_batches
                    assert self.on_train_index == (j + 1 + k) * self.batch_size % self.n_train
            if vis:
                from utils import save_visualizations
                save_visualizations(images, self.data_dir)
                import ipdb; ipdb.set_trace()

                #assert self.on_epoch == i + 1
        except AssertionError as e:
            print(e)
            import ipdb; ipdb.set_trace()

class MNIST(LabelledDataset):
    def __init__(self, batch_size, data_dir, dims, one_hot, unbalanced_train=False, unbalanced_test=False):
        name = 'MNIST'
        if unbalanced_train:
            train_tag = "/train-{}".format(unbalanced_train)
            name += train_tag
        if unbalanced_test:
            test_tag = "/test-{}".format(unbalanced_test)
            name += test_tag
        self.unbalanced_test = unbalanced_test
        self.unbalanced_train = unbalanced_train

        super(MNIST, self).__init__(batch_size, data_dir + '/MNIST', name, dims, [28, 28, 1], 10, one_hot)

    def load_train_data(self, data_dir):
        data_dir_ = data_dir
        if self.unbalanced_train:
            data_dir_ += '/u-{}'.format(self.unbalanced_train)
        self.x_train = np.load(os.path.join(data_dir_, 'x-train.npy'))
        self.y_train = np.load(os.path.join(data_dir_, 'y-train.npy'))

    def load_test_data(self, data_dir):
        data_dir_ = data_dir
        if self.unbalanced_test:
            data_dir_ += '/u-{}'.format(self.unbalanced_test)
        self.x_test = np.load(os.path.join(data_dir_, 'x-test.npy'))
        self.y_test = np.load(os.path.join(data_dir_, 'y-test.npy'))

    def x_preprocess(self, x):
        return np.reshape(x, [-1, self.input_dim]) / 255.0


class CIFAR10(LabelledDataset):
    def __init__(self, batch_size, data_dir, one_hot):
        name = 'cifar10'
        super(CIFAR10, self).__init__(batch_size, data_dir + '/cifar10', name, 4, [32, 32, 3], 10, one_hot)

    def x_preprocess(self, x):
        reshape = np.reshape(x, (x.shape[0], self.input_shape[2], self.input_shape[0], self.input_shape[1])) / 255.0
        x_T = np.transpose(reshape, (0, 2, 3, 1))
        return x_T

class CIFAR10_u05(LabelledDataset):
    def __init__(self, batch_size, data_dir, one_hot):
        name = 'cifar10/u-0.5'
        super(CIFAR10_u05, self).__init__(batch_size, os.path.join(data_dir, name), name, 4, [32, 32, 3], 10, one_hot)

    def x_preprocess(self, x):
        x_ = np.transpose(x, (0, 2, 3, 1))
        reshape = np.reshape(x_, (x_.shape[0], *self.input_shape)) / 255.0
        return reshape

def get_labeled_data(imagefile, labelfile):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    # Open the images with gzip in read binary mode
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = zeros((N, rows, cols), dtype=float32)  # Initialize numpy array
    y = zeros((N, 1), dtype=uint8)  # Initialize numpy array
    for i in range(N):
        if i % 1000 == 0:
            print("i: %i" % i)
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (x, y)

def make_unbalanced(data_dir, keep_prob=0.5):
    (x_train, y_train), (x_test, y_test) = load_from_standard_supervised_format(data_dir)

    def sample(keep_prob, x, y):
        p = np.exp(y * np.log(keep_prob))
        coins = np.random.binomial(1, p)
        select = np.nonzero(coins)[0]
        x_out = x[select]
        y_out = y[select]
        return x_out, y_out

    x_train_u, y_train_u = sample(keep_prob, x_train, y_train)
    x_test_u, y_test_u = sample(keep_prob, x_test, y_test)

    path = os.path.join(data_dir, 'u-{}'.format(keep_prob))
    if not os.path.exists(path):
        os.makedirs(path)
    save_to_standard_supervised_format(x_train_u, y_train_u, x_test_u, y_test_u, path)
#make_unbalanced_mnist()

def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

def save_to_standard_supervised_format(x_train, y_train, x_test, y_test, data_dir):
    np.save(os.path.join(data_dir, 'x-train.npy'), x_train)
    np.save(os.path.join(data_dir, 'y-train.npy'), y_train)
    np.save(os.path.join(data_dir, 'x-test.npy'), x_test)
    np.save(os.path.join(data_dir, 'y-test.npy'), y_test)

def load_from_standard_supervised_format(data_dir):
    x_train = np.load(os.path.join(data_dir, 'x-train.npy'), 'r')
    y_train = np.load(os.path.join(data_dir, 'y-train.npy'), 'r')
    x_test = np.load(os.path.join(data_dir, 'x-test.npy'), 'r')
    y_test = np.load(os.path.join(data_dir, 'y-test.npy'), 'r')
    return (x_train, y_train), (x_test, y_test)

def make_cifar10():
    f_dicts = []
    for i in range(1, 6):
        f_dicts.append(unpickle(os.path.join(DATA_DIR, 'cifar10/cifar-10-batches-py/data_batch_{}'.format(i))))
    x_train = np.concatenate([f_dict['data'] for f_dict in f_dicts])
    y_train = np.concatenate([f_dict['labels'] for f_dict in f_dicts])

#make_unbalanced(DATA_DIR + '/cifar10')
