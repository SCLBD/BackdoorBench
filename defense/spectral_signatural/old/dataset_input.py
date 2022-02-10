"""
Utilities for importing the CIFAR10 dataset.

Each image in the dataset is a numpy array of shape (32, 32, 3), with the values
being unsigned integers (i.e., in the range 0,1,...,255).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import random
import os
import sys
import tensorflow as tf
version = sys.version_info

import numpy as np
import scipy.io as sio
from functools import reduce

def poison(x, method, pos, col):
    ret_x = np.copy(x)
    col_arr = np.asarray(col)
    if ret_x.ndim == 3:
        #only one image was passed
        if method=='pixel':
            ret_x[pos[0],pos[1],:] = col_arr
        elif method=='pattern':
            ret_x[pos[0],pos[1],:] = col_arr
            ret_x[pos[0]+1,pos[1]+1,:] = col_arr
            ret_x[pos[0]-1,pos[1]+1,:] = col_arr
            ret_x[pos[0]+1,pos[1]-1,:] = col_arr
            ret_x[pos[0]-1,pos[1]-1,:] = col_arr
        elif method=='ell':
            ret_x[pos[0], pos[1],:] = col_arr
            ret_x[pos[0]+1, pos[1],:] = col_arr
            ret_x[pos[0], pos[1]+1,:] = col_arr
    else:
        #batch was passed
        if method=='pixel':
            ret_x[:,pos[0],pos[1],:] = col_arr
        elif method=='pattern':
            ret_x[:,pos[0],pos[1],:] = col_arr
            ret_x[:,pos[0]+1,pos[1]+1,:] = col_arr
            ret_x[:,pos[0]-1,pos[1]+1,:] = col_arr
            ret_x[:,pos[0]+1,pos[1]-1,:] = col_arr
            ret_x[:,pos[0]-1,pos[1]-1,:] = col_arr
        elif method=='ell':
            ret_x[:,pos[0], pos[1],:] = col_arr
            ret_x[:,pos[0]+1, pos[1],:] = col_arr
            ret_x[:,pos[0], pos[1]+1,:] = col_arr
    return ret_x

class CIFAR10Data(object):
    """
    Unpickles the CIFAR10 dataset from a specified folder containing a pickled
    version following the format of Krizhevsky which can be found
    [here](https://www.cs.toronto.edu/~kriz/cifar.html).

    Inputs to constructor
    =====================

        - path: path to the pickled dataset. The training data must be pickled
        into five files named data_batch_i for i = 1, ..., 5, containing 10,000
        examples each, the test data
        must be pickled into a single file called test_batch containing 10,000
        examples, and the 10 class names must be
        pickled into a file called batches.meta. The pickled examples should
        be stored as a tuple of two objects: an array of 10,000 32x32x3-shaped
        arrays, and an array of their 10,000 true labels.

    """
    def __init__(self, config, seed=None):
        train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]
        eval_filename = 'test_batch'
        metadata_filename = 'batches.meta'
        self.rng = np.random.RandomState(1) if seed is None else np.random.RandomState(seed)

        model_dir = config.model.output_dir
        path = config.data.cifar10_path
        method = config.data.poison_method
        eps = config.data.poison_eps
        clean = config.data.clean_label
        target = config.data.target_label
        position = config.data.position
        color = config.data.color
        num_training_examples = config.training.num_examples

        train_images = np.zeros((50000, 32, 32, 3), dtype='uint8')
        train_labels = np.zeros(50000, dtype='int32')
        for ii, fname in enumerate(train_filenames):
            cur_images, cur_labels = self._load_datafile(os.path.join(path, fname))
            train_images[ii * 10000 : (ii+1) * 10000, ...] = cur_images
            train_labels[ii * 10000 : (ii+1) * 10000, ...] = cur_labels
        eval_images, eval_labels = self._load_datafile(
            os.path.join(path, eval_filename))

        if eps>0:
            if clean>-1:
                clean_indices = np.where(train_labels==clean)[0]
            else:
                clean_indices = np.where(train_labels!=target)[0]
            poison_indices = self.rng.choice(clean_indices, eps, replace=False)
            poison_images = np.zeros((eps, 32, 32, 3))
            for i in range(eps):
                poison_images[i] = poison(train_images[poison_indices[i]], method, position, color)
            train_images = np.concatenate((train_images, poison_images), axis=0)
            if target>-1:
                poison_labels = np.repeat(target, eps)
            else:
                poison_labels = self.rng.randint(0,10, eps)
            train_labels = np.concatenate((train_labels, poison_labels), axis=0)
            train_images = np.delete(train_images, poison_indices, axis=0)
            train_labels = np.delete(train_labels, poison_indices, axis=0)

        train_indices = np.arange(len(train_images))
        eval_indices = np.arange(len(eval_images))
        
        with open(os.path.join(path, metadata_filename), 'rb') as fo:
              if version.major == 3:
                  data_dict = pickle.load(fo, encoding='bytes')
              else:
                  data_dict = pickle.load(fo)

              self.label_names = data_dict[b'label_names']
        for ii in range(len(self.label_names)):
            self.label_names[ii] = self.label_names[ii].decode('utf-8')

        removed_indices_file = os.path.join(model_dir, 'removed_inds.npy')
        if os.path.exists(removed_indices_file):
            removed = np.load(os.path.join(model_dir, 'removed_inds.npy'))
            train_indices = np.delete(train_indices, removed)

        self.num_poisoned_left = np.count_nonzero(train_indices>=(50000-eps))
        #for debugging purpos
        np.save(os.path.join(model_dir, 'train_indices.npy'), train_indices)
        poisoned_eval_images = poison(eval_images, method, position, color)

        if config.model.per_im_std:
            train_images = self._per_im_std(train_images)
            eval_images = self._per_im_std(eval_images)
            poisoned_eval_images = self._per_im_std(poisoned_eval_images)
            
        self.train_data = DataSubset(train_images[train_indices], train_labels[train_indices])
        self.eval_data = DataSubset(eval_images[eval_indices], eval_labels[eval_indices])
        self.poisoned_eval_data = DataSubset(poisoned_eval_images[eval_indices], eval_labels[eval_indices])

    @staticmethod
    def _load_datafile(filename):
        with open(filename, 'rb') as fo:
            if version.major == 3:
                data_dict = pickle.load(fo, encoding='bytes')
            else:
                data_dict = pickle.load(fo)

            assert data_dict[b'data'].dtype == np.uint8
            image_data = data_dict[b'data']
            image_data = image_data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
            return image_data, np.array(data_dict[b'labels'])

    @staticmethod      
    def _per_im_std(ims):
        split_ims = np.split(ims, ims.shape[0], axis=0)
        num_pixels = reduce(lambda x,y:x*y, list(split_ims[0].shape),1)
        for ii in range(len(split_ims)):
            curmean = np.mean(split_ims[ii],keepdims=True)
            split_ims[ii] = split_ims[ii] - curmean
            curstd = np.std(split_ims[ii],keepdims=False)
            adjustedstd = max(curstd, 1.0/np.sqrt(num_pixels))
            split_ims[ii] = split_ims[ii]/adjustedstd
        return np.concatenate(split_ims)

class DataSubset(object):
    def __init__(self, xs, ys, num_examples=None, seed=None):
        self.rng = np.random.RandomState(1) if seed is None \
                   else np.random.RandomState(seed)
        if num_examples:
            xs, ys = self._per_class_subsample(xs, ys, num_examples,
                                               rng=self.rng)
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.n)

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        self.batch_start += actual_batch_size
        return batch_xs, batch_ys

