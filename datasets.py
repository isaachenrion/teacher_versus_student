from collections import namedtuple

from struct import unpack
import gzip
from numpy import zeros, uint8, float32
import os
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.examples.tutorials.mnist import input_data

DATASETS_DIR = 'data'
DATASET_TYPES = ['train', 'valid', 'test']
def create_binarized_mnist(which):
    with open(DATASETS_DIR + '/BinaryMNIST/binarized_mnist_{}.amat'.format(which)) as f:
        data = [l.strip().split(' ') for l in f.readlines()]
        data = np.array(data).astype(int)
        np.save(DATASETS_DIR + '/BinaryMNIST/binarized_mnist_{}.npy'.format(which), data)
    return data

def binarized_mnist():
    #for which in ['train', 'valid', 'test']
    #    data = np.load(DATASETS_DIR + '/BinaryMNIST/binarized_mnist_{}.npy'.format(which))
    #    dataset = UnlabelledDataSet(data)
    dataset = {which: UnlabelledDataSet(np.load(DATASETS_DIR + '/BinaryMNIST/binarized_mnist_{}.npy'.format(which)))for which in DATASET_TYPES}
    return dataset

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
    def __init__(self, batch_size, data_dir, name, dims, input_shape, label_dim, one_hot):
        self.data_dir = name
        self.name = name
        self.dims = dims
        self.one_hot = one_hot

        self.x_train = np.load(os.path.join(data_dir, 'x-train.npy'))
        self.y_train = np.load(os.path.join(data_dir, 'y-train.npy'))
        self.x_test = np.load(os.path.join(data_dir, 'x-test.npy'))
        self.y_test = np.load(os.path.join(data_dir, 'y-test.npy'))

        self.input_shape = input_shape
        self.input_dim = np.prod(self.input_shape)
        self.label_dim = label_dim

        self.reset_counters()
        self.batch_size = batch_size
        self.n_train_batches = self.x_train.shape[0] / batch_size
        self.n_test_batches = self.x_test.shape[0] / batch_size

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
        self.n_train_batches = self.x_train.shape[0] / batch_size
        self.n_test_batches = self.x_test.shape[0] / batch_size

    def get_batch(self, test=False, random=False):
        if test:
            return self.get_test_batch(random)
        else:
            return self.get_train_batch(random)

    def get_train_batch(self, random=False):
        if self.on_train_batch > self.n_train_batches:
            self.on_train_index = 0
            self.on_train_batch = 0
            self.on_epoch += 1
        start, end = self.on_train_index, self.on_train_index + self.batch_size
        self.on_train_batch += 1
        self.on_train_index += self.batch_size

        return self.x_preprocess(self.x_train[start:end]), self.y_preprocess(self.y_train[start:end])

    def get_test_batch(self, random=False):
        if self.on_test_batch > self.n_test_batches:
            self.on_test_index = 0
            self.on_test_batch = 0
            self.on_epoch += 1
        start, end = self.on_test_index, self.on_test_index + self.batch_size
        self.on_test_batch += 1
        self.on_test_index += self.batch_size

        return self.x_preprocess(self.x_test[start:end]), self.y_preprocess(self.y_test[start:end])

def mnist(batch_size, data_directory):
    return LabelledDataset(batch_size, data_directory, name='mnist', dims=4, one_hot=True, input_shape=[28, 28, 1], label_dim=10)

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
