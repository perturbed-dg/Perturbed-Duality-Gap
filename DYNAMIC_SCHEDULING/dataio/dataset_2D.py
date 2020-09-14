import os, sys
import numpy as np
import random
import tensorflow as tf

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_path)

from dataio.dataset import Dataset
import tensorflow_datasets as tfds
import sklearn
import json
from sklearn import datasets

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def gaussian_8_modes(DATASET_SIZE):
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]
    while True:
        dataset = []
        for i in range(DATASET_SIZE):
            point = np.random.randn(2) * .02
            center = centers[np.random.choice(len(centers))]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        dataset /= 1.414  
        return dataset,np.ones(dataset.shape[0])
      
def gaussian_9_modes(DATASET_SIZE):
    dataset = []
    for i in range(int((DATASET_SIZE / 9))):
        for x in range(-3, 3,2):
            for y in range(-3, 3,2):
                point = np.random.randn(2) * 0.05
                point[0] += 2 * x
                point[1] += 2 * y
                dataset.append(point)
    dataset = np.array(dataset, dtype='float32')
    np.random.shuffle(dataset)
    dataset /= 2.828  

    return dataset,np.ones(dataset.shape[0])
    
def swiss_roll(DATASET_SIZE):
    batch_data = sklearn.datasets.make_swiss_roll(n_samples=DATASET_SIZE, noise=0.25)[0]
    batch_data = batch_data.astype(np.float32)[:, [0, 2]]
    batch_data /= 7.5
    return batch_data,np.ones(batch_data.shape[0])
    
def load_data(dataset,DATASET_SIZE=90000):

    if(dataset == 'ring'):
        return gaussian_8_modes(DATASET_SIZE)
    elif(dataset == 'grid'):
        return gaussian_9_modes(DATASET_SIZE)
    elif(dataset=='spiral'):
        return swiss_roll(DATASET_SIZE)
    else:
        print('Invalid Dataset')
        exit(0)

class Dataset_2D(Dataset):
    def __init__(self):
        pass

    def load_data(self, data_task='grid',data_dir='', mode=tf.estimator.ModeKeys.TRAIN):
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            self._dataset_input,self._dataset_target = load_data(data_task)
        else:
            pass
        
        self._num_examples = len(self._dataset_target)
        print('Load {} samples.'.format(self._num_examples))
        self._index = np.arange(self._num_examples)
        self._index_in_epoch = 0
        self._epochs_completed = 0


if __name__ == '__main__':
    dataset = Dataset_2D()
    dataset.load_data(data_task='ring',mode=tf.estimator.ModeKeys.TRAIN)
