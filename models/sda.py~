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

import cPickle
import gzip
import numpy
import os
import sys
import theano
from theano.tensor.shared_randomstreams import RandomStreams
import time

from dlnn.layers.da import dA, dA_maxout
from dlnn.layers.logistic_sgd import LogisticRegression
from dlnn.layers.mlp import HiddenLayer
import theano.tensor as T
from dlnn.utils.utils import shared_dataset_X, shared_dataset
from dlnn.io_func.model_io import _nnet2file, _cfg2file, _file2nnet, log
from sklearn.preprocessing import OneHotEncoder

class SdA(object):

    def __init__(self, numpy_rng, theano_rng=None, cfg = None, dnn = None):
        """ Stacked Denoising Autoencoders for DNN Pre-training """

        self.cfg = cfg
        self.hidden_layers_sizes = cfg.hidden_layers_sizes
        self.n_ins = cfg.n_ins
        self.hidden_layers_number = len(self.hidden_layers_sizes)

        self.dA_layers = []
        self.sigmoid_layers = []
        self.params = []
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = dnn.x

        for i in xrange(self.hidden_layers_number):
            # the size of the input is either the number of hidden units of
            # the layer below, or the input size if we are on the first layer
            if i == 0:
                input_size = self.n_ins
                layer_input = self.x
            else:
                input_size = self.hidden_layers_sizes[i - 1]
                layer_input = dnn.layers[i-1].output

            # Construct a denoising autoencoder that shared weights with this layer
            if i == 0:
                reconstruct_activation = cfg.firstlayer_reconstruct_activation
            else:
                reconstruct_activation = cfg.hidden_activation
            self.sigmoid_layers.append(dnn.layers[i])
            self.params.extend(dnn.layers[i].params)
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=self.hidden_layers_sizes[i],
                          W=dnn.layers[i].W,
                          bhid=dnn.layers[i].b,
                          sparsity = cfg.sparsity,
                          sparsity_weight = cfg.sparsity_weight,
                          hidden_activation = cfg.hidden_activation,
                          reconstruct_activation = reconstruct_activation,
                          contraction_level= self.cfg.contraction_level, 
                          n_batchsize = self.cfg.batch_size)
            self.dA_layers.append(dA_layer)
    def get_cost_functions(self, train_set_x):
        # get cost of train_set_x or validation set for each dA layer.
        cost_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(0, 0)
            # compile the theano function
            fn = theano.function(inputs=[],
                                 outputs=cost,
                                 givens={self.x: train_set_x})
            # append `fn` to the list of functions
            cost_fns.append(fn)
    def pretraining_functions(self, train_set_x, batch_size):

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        momentum = T.scalar('momentum')
        batch_size = batch_size if train_set_x.get_value(borrow=True).shape[0] > batch_size else train_set_x.get_value(borrow=True).shape[0]
        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level, learning_rate, momentum)
            # compile the theano function
            fn = theano.function(inputs=[index,
                              theano.Param(corruption_level, default=0.2),
                              theano.Param(learning_rate, default=0.1)
                              ,theano.Param(momentum, default=0.5)
                              ],
                              outputs=cost,
                              updates=updates,
                              givens={self.x: train_set_x[batch_begin:batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)
        return pretrain_fns
    
    # pretraining with early stop.    
    def pretraining_with_estop(self, X_train_minmax, settings):
            batch_size = settings['batch_size']
            corruption_levels = settings['corruption_levels']
            pretrain_lr = settings['pretrain_lr']
            momentum = settings['momentum']
            pretraining_epochs = settings['pretraining_epochs']
            n_visible = X_train_minmax.shape[1]
            # shuffle examples:
            from sklearn.utils import shuffle
            X_train_minmax = shuffle(X_train_minmax, random_state=0)
            
            train_set_x = shared_dataset_X(X_train_minmax[ :X_train_minmax.shape[0] / 2, :], borrow=True)
            valid_set_x = shared_dataset_X(X_train_minmax[ X_train_minmax.shape[0] / 2:, :], borrow=True)
            # compute number of minibatches for training, validation and testing

            n_train_batches = train_set_x.get_value(borrow=True).shape[0]

            if n_train_batches <= batch_size:
                batch_size = n_train_batches
            n_train_batches /= batch_size
            # numpy random generator
            numpy_rng = numpy.random.RandomState(66)
            validation_funcs = self.get_cost_functions(valid_set_x)
            pretraining_fns = self.pretraining_functions(train_set_x, batch_size)
            # early-stopping parameters

            # patience = 1000 * n_train_batches 
            patience_increase = 2.  # wait this much longer when a new best is
                                    # found
            improvement_threshold = 0.995  # a relative improvement of this much is
                                           # considered significant
            validation_frequency = min(n_train_batches * pretraining_epochs / 200, 100)
                                          # go through this many
                                          # minibatche before checking the network
                                          # on the validation set; in this case we
                                          # check every epoch
            print '... pre-training the model'
            start_time = time.clock()
            # Pre-train layer-wise
            for i in xrange(self.n_layers):
                # go through pretraining epochs
                best_params = None
                best_validation_loss = numpy.inf
                test_score = 0.
                done_looping = False
                epoch = 0
                best_iter = 0
                patience = 5000  # look as this many iterations
                while epoch < pretraining_epochs and (not done_looping):
                    # go through the training set
                    c = []
                    for minibatch_index in xrange(n_train_batches):
                        c.append(pretraining_fns[i](index=minibatch_index,
                                 corruption=corruption_levels[i],
                                 lr=pretrain_lr, momentum = momentum))
                        iter = epoch * n_train_batches + minibatch_index +1
                        if (iter + 1) % validation_frequency == 0:
                                
                                this_validation_fn = validation_funcs[i]
                                this_validation_loss = this_validation_fn()
                                print('epoch %i, minibatch %i/%i, validation cost %f ' % 
                                      (epoch, minibatch_index + 1, n_train_batches,
                                       this_validation_loss))

                                # if we got the best validation score until now
                                if this_validation_loss < best_validation_loss:

                                    # improve patience if loss improvement is good enough
                                    if (this_validation_loss < best_validation_loss * 
                                        improvement_threshold):
                                        patience = max(patience, iter * patience_increase)

                                    # save best validation score and iteration number
                                        best_validation_loss = this_validation_loss
                                        best_iter = iter
                        if patience <= iter:
                                done_looping = True
                                break    
                    print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                    print numpy.mean(c)
                    epoch += 1
            end_time = time.clock()

            print >> sys.stderr, ('The pretraining code ran for %.2fm' % ((end_time - start_time) / 60.))
            
    # pretraining without early stop.        
    def pretraining(self, train_x = None):
            log('> ... getting the pre-training functions')
            if train_x is not None:
                    batch_size = self.cfg.batch_size if train_x.shape[0] > self.cfg.batch_size else train_x.shape[0]
                    pretraining_fns = self.pretraining_functions(train_set_x=shared_dataset_X(train_x), batch_size=batch_size)

                    # resume training
                    start_layer_index = 0; start_epoch_index = 

                    log('> ... pre-training the model')
                    # layer by layer; for each layer, go through the epochs
                    for i in range(start_layer_index, self.cfg.ptr_layer_number):
                        for epoch in range(start_epoch_index, self.cfg.epochs):
                            # go through the training set
                                c = []

                                for batch_index in xrange(train_x.shape[0] / batch_size):  # loop over mini-batches
                                    c.append(pretraining_fns[i](index=batch_index,
                                                                corruption=self.cfg.corruption_levels[i],
                                                                lr=self.cfg.learning_rates[i]
                                                                , momentum=self.cfg.momentum
                                                                ))
                                
                                log('> layer %i, epoch %d, reconstruction cost %f' % (i, epoch, numpy.mean(c)))
    # transform the input data to the last layer activations   
    def transform(self, data_x):  # get the last layer activations to transform data.
        last_layer_activations = self.sigmoid_layers[-1].output
        theano_fn = theano.function(inputs=[],
                                 outputs=last_layer_activations,

                                 givens={self.x: data_x})
        newFeatures = theano_fn()
        return newFeatures
    
class Sda_xy(SdA):
    '''
    Stacked target-aware auto encoders. This class is an extension of Sda. It dose the unsupervised training by reconstruction of XY. In order to avoid the overwhelming effect of X, we will give more weight on Y when calculating the reconstruction error. 
    '''
    def __init__(self, numpy_rng, theano_rng=None, cfg = None):
        
        self.cfg = cfg
        self.hidden_layers_sizes = cfg.hidden_layers_sizes
        self.n_ins = cfg.n_ins+ cfg.n_outs
        self.hidden_layers_number = len(self.hidden_layers_sizes)

        self.dA_layers = []
        self.sigmoid_layers = []
        self.params = []
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.xy = T.matrix('xy', dtype=theano.config.floatX) 

        for i in xrange(self.hidden_layers_number):
            # the size of the input is either the number of hidden units of
            # the layer below, or the input size if we are on the first layer
            
            if i == 0:
                input_size = self.n_ins

            else:
                if self.cfg.settings.has_key('firstlayer_xy') and self.cfg.settings['firstlayer_xy'] ==  1:
                    input_size = self.hidden_layers_sizes[i - 1]
                else:
                    input_size = self.hidden_layers_sizes[i - 1] + cfg.n_outs


            # Construct a denoising autoencoder that shared weights with this layer
            if i == 0:
                reconstruct_activation = cfg.firstlayer_reconstruct_activation
            else:
                reconstruct_activation = cfg.hidden_activation

            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=None,
                          n_visible=input_size,
                          n_hidden=self.hidden_layers_sizes[i],
                          sparsity = cfg.sparsity,
                          sparsity_weight = cfg.sparsity_weight,
                          hidden_activation = cfg.hidden_activation,
                          reconstruct_activation = reconstruct_activation,
                          contraction_level= self.cfg.contraction_level,
                          n_batchsize = self.cfg.batch_size)
            self.dA_layers.append(dA_layer)
    def pretraining_function(self, dA, train_set_x, batch_size):

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        momentum = T.scalar('momentum')
        # number of batches
        batch_size = batch_size if train_set_x.get_value(borrow=True).shape[0] > batch_size else train_set_x.get_value(borrow=True).shape[0]
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size


        cost, updates = dA.get_cost_updates(corruption_level, learning_rate, momentum, weight_y = self.cfg.settings['weight_y'] , size_of_x = self.cfg.n_ins, size_of_y = self.cfg.n_outs)
        # compile the theano function
        fn = theano.function(inputs=[index,
                      theano.Param(corruption_level, default=0.2),
                      theano.Param(learning_rate, default=0.1)
                      ,theano.Param(momentum, default=0.5)
                      ],
                      outputs=cost,
                      updates=updates,
                      givens={dA.x: train_set_x[batch_begin:batch_end]})

        return fn
    def pretraining(self, train_x = None, train_y = None):
            #print len(numpy.shape(train_y))
            
            multi_classes = True if self.cfg.n_outs >= 3 else False
            train_y  = train_y.astype(dtype = theano.config.floatX)
            train_y_T = train_y[numpy.newaxis].T
            
            if multi_classes == False:               
                train_xy = numpy.hstack((train_x, train_y_T))
                shared_train_xy = shared_dataset_X(train_xy) 
            else:
                enc = OneHotEncoder(n_values = self.cfg.n_outs, dtype = theano.config.floatX, sparse=False)
                encode_train_y = enc.fit_transform(train_y_T)
                shared_train_xy = shared_dataset_X(numpy.hstack((train_x, encode_train_y)))              
            log('> ... getting the pre-training functions')
            if train_x is None: # this means we are using the stream input from file
                pass
            else: # this means using numpy matrix as input

                    start_layer_index = 0; start_epoch_index = 0

                    log('> ... pre-training the model')
                    # layer by layer; for each layer, go through the epochs
                    for i in range(start_layer_index, self.cfg.ptr_layer_number):
                        pretraining_fn = self.pretraining_function(self.dA_layers[i], train_set_x = shared_train_xy, batch_size = self.cfg.batch_size)
                        for epoch in range(start_epoch_index, self.cfg.epochs):
                            # go through the training set
                                c = []
#                            while (not self.cfg.train_sets.is_finish()):
#                                self.cfg.train_sets.load_next_partition(self.cfg.train_xy)
                                iteration_per_epoch = train_x.shape[0] / self.cfg.batch_size if train_x.shape[0] / self.cfg.batch_size else 1
                                for batch_index in xrange(iteration_per_epoch):  # loop over mini-batches
                                    c.append(pretraining_fn(index=batch_index,
                                                                corruption=self.cfg.corruption_levels[i],
                                                                lr=self.cfg.learning_rates[i]
                                                                , momentum=self.cfg.momentum
                                                                ))
#                            self.cfg.train_sets.initialize_read()
                                log('> layer %i, epoch %d, reconstruction cost %f' % (i, epoch, numpy.mean(c)))  
                        hidden_values = self.dA_layers[i].transform(shared_train_xy.get_value())
                        if self.cfg.settings.has_key('firstlayer_xy') and self.cfg.settings['firstlayer_xy'] ==  1:
                            shared_train_xy = shared_dataset_X(hidden_values)
                        else: # add y for every layer 
                            if multi_classes == False:               
                                train_xy = numpy.hstack((hidden_values, train_y_T))
                                shared_train_xy = shared_dataset_X(train_xy) 
                            else:
                                shared_train_xy = shared_dataset_X(numpy.hstack((hidden_values, encode_train_y)))   
# an outdated class for SdA with maxout hidden activation. pre-training has been expirically found to be
# NOT helpful for maxout networks, so we don't update this class
class SdA_maxout(object):

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10,
                 corruption_levels=[0.1, 0.1], pool_size = 3,
                 sparsity = None, sparsity_weight = None,
                 first_reconstruct_activation = T.tanh):

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i] * pool_size,
                                        activation=(lambda x: 1.0*x),
                                        do_maxout = True, pool_size = pool_size)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this layer
            if i == 0:
                reconstruct_activation = first_reconstruct_activation
            else:
                reconstruct_activation = (lambda x: 1.0*x)
#               reconstruct_activation = first_reconstruct_activation
            dA_layer = dA_maxout(numpy_rng=numpy_rng,
                              theano_rng=theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=hidden_layers_sizes[i] * pool_size,
                              W=sigmoid_layer.W,
                              bhid=sigmoid_layer.b,
                              sparsity = sparsity,
                              sparsity_weight = sparsity_weight,
                              pool_size = pool_size,
                              reconstruct_activation = reconstruct_activation)
            self.dA_layers.append(dA_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
                         input=self.sigmoid_layers[-1].output,
                         n_in=hidden_layers_sizes[-1], n_out=n_outs)

        self.sigmoid_layers.append(self.logLayer)
        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

    def pretraining_functions(self, train_set_x, batch_size):

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(inputs=[index,
                              theano.Param(corruption_level, default=0.2),
                              theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x: train_set_x[batch_begin:
                                                             batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns
