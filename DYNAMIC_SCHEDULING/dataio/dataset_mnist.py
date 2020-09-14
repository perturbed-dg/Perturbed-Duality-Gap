import os, sys
import numpy as np
import random
import tensorflow as tf

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_path)
from dataio.dataset import Dataset
# from tensorflow.examples.tutorials.mnist import input_data
import tensorflow_datasets as tfds

def normalize_image(ex):
  ex['image'] = tf.to_float(ex['image']) / 255.
  return ex

class Dataset_mnist(Dataset):
    def __init__(self):
        pass

    def load_mnist(self, data_dir, mode=tf.estimator.ModeKeys.TRAIN):
        # mnist = input_data.read_data_sets(data_dir, one_hot=True)
        if mode == tf.estimator.ModeKeys.TRAIN:
            # _dataset = mnist.train
            # _dataset = tfds.load(name="mnist", split=tfds.Split.TRAIN)
            self._dataset_input,self._dataset_target = tfds.as_numpy(tfds.load(
                'mnist',
                split='train', 
                batch_size=-1, 
                as_supervised=True,
            ))
            self._dataset_input = np.array((self._dataset_input)/255,dtype='float32')
        else:
            # _dataset = mnist.test
            # _dataset = tfds.load(name="mnist", split=tfds.Split.TEST)
            self._dataset_input , self._dataset_target =  tfds.as_numpy(tfds.load(
                'mnist',
                split='test', 
                batch_size=-1, 
                as_supervised=True,
            ))
            self._dataset_input = np.array((self._dataset_input)/255,dtype='float32')
        self._num_examples = len(self._dataset_target)
        print('Load {} samples.'.format(self._num_examples))
        self._index = np.arange(self._num_examples)
        self._index_in_epoch = 0
        self._epochs_completed = 0


if __name__ == '__main__':
    dataset = Dataset_mnist()
    dataset.load_mnist(mode=tf.estimator.ModeKeys.TRAIN)
