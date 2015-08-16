# Copyright 2015    Tianchuan Du    University of Delaware

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import theano.tensor as T
from dlnn.utils.learn_rates import LearningRateConstant, LearningRateExpDecay


# validation on the valid data; this involves a forward pass of all the valid data into the network,
# mini-batch by mini-batch
# valid_fn: the compiled valid function
# valid_sets: the dataset object for valid
# valid_xy: the tensor variables for valid dataset
# batch_size: the size of mini-batch
# return: a list containing the *error rates* on each mini-batch
def validate_by_minibatch_verbose(valid_fn, valid_sets, valid_xy, batch_size):
    valid_error = []
    while (not valid_sets.is_finish()):
        valid_sets.load_next_partition(valid_xy)
        for batch_index in xrange(valid_sets.cur_frame_num / batch_size):  # loop over mini-batches
            valid_error.append(valid_fn(index=batch_index))
    valid_sets.initialize_read()
    return valid_error

def validate_by_minibatch(valid_fn, cfg):
    valid_sets = cfg.valid_sets; valid_xy = cfg.valid_xy
    batch_size = cfg.batch_size
    valid_error = []
    while (not valid_sets.is_finish()):
        valid_sets.load_next_partition(valid_xy)
        for batch_index in xrange(valid_sets.cur_frame_num / batch_size):  # loop over mini-batches
            valid_error.append(valid_fn(index=batch_index))
    valid_sets.initialize_read()
    return valid_error
def validate_by_minibatch_without_streaming(valid_fn, n_valid, cfg):
    
    batch_size = cfg.batch_size
    valid_error = []
    n_valid = n_valid
    for batch_index in xrange(n_valid / batch_size):  # loop over mini-batches
        valid_error.append(valid_fn(index=batch_index))

    return valid_error

# one epoch of mini-batch based SGD on the training data
# train_fn: the compiled training function
# train_sets: the dataset object for training
# train_xy: the tensor variables for training dataset
# batch_size: the size of mini-batch
# learning_rate: learning rate
# momentum: momentum
# return: a list containing the *error rates* on each mini-batch
def train_sgd_verbose(train_fn, train_sets, train_xy, batch_size, learning_rate, momentum):
    train_error = []
    while (not train_sets.is_finish()):
        train_sets.load_next_partition(train_xy)
        for batch_index in xrange(train_sets.cur_frame_num / batch_size):  # loop over mini-batches
            train_error.append(train_fn(index=batch_index, learning_rate = learning_rate, momentum = momentum))
    train_sets.initialize_read()
    return train_error

def train_sgd(train_fn, cfg):
    train_sets = cfg.train_sets; train_xy = cfg.train_xy
    batch_size = cfg.batch_size
    learning_rate = cfg.lrate.get_rate(); momentum = cfg.momentum 
    
    train_error = []
    while (not train_sets.is_finish()):
        train_sets.load_next_partition(train_xy)
        for batch_index in xrange(train_sets.cur_frame_num / batch_size):  # loop over mini-batches
            train_error.append(train_fn(index=batch_index, learning_rate = learning_rate, momentum = momentum))
    train_sets.initialize_read()
    return train_error
def train_sgd_without_streaming(train_fn, n_train, cfg):
    # this means the data is not loading form streaming file system
    batch_size = cfg.batch_size
    learning_rate = cfg.lrate.get_rate(); momentum = cfg.momentum 
    train_error = []
    n_train = n_train
    for batch_index in xrange(n_train / batch_size):  # loop over mini-batches
        train_error.append(train_fn(index=batch_index, learning_rate = learning_rate, momentum = momentum))
    return train_error
