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
import collections
import gzip
import numpy
import os
import sys
import theano
from theano.tensor.shared_randomstreams import RandomStreams
import time

import theano.tensor as T
from numpy import dtype


class dA(object):
    """Denoising Auto-Encoder class (dA)"""

    def __init__(self, numpy_rng, theano_rng=None, input=None,
                 n_visible=500, n_hidden=500,
                 W=None, bhid=None, bvis=None, sparsity = None,
                 sparsity_weight = None,
                 hidden_activation = T.nnet.sigmoid,
                 reconstruct_activation = T.nnet.sigmoid,
                 contraction_level = 0,
                 n_batchsize = 1):

        self.type = 'fc'

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_batchsize = n_batchsize

        self.sparsity = sparsity
        self.sparsity_weight = sparsity_weight
        self.contraction_level = contraction_level
        
        self.hidden_activation = hidden_activation
        self.reconstruct_activation = reconstruct_activation
        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng 

        # if no input is given, generate a variable representing the input
        if input == None:
#            self.x = T.dmatrix(name='input')
            self.x = T.matrix(name='input', dtype=theano.config.floatX)
        else:
            self.x = input

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            if hidden_activation == T.nnet.sigmoid:
                initial_W *= 4
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible, dtype=theano.config.floatX),borrow=True)

        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden, dtype=theano.config.floatX), name='b', borrow=True)

        self.W = W
        # b -- the bias of the hidden
        self.b = bhid
        # b_prime -- the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T

        self.delta_W = theano.shared(value=numpy.zeros_like(W.get_value(borrow=True), dtype=theano.config.floatX), name='delta_W')
        self.delta_b = theano.shared(value=numpy.zeros_like(bhid.get_value(borrow=True), dtype=theano.config.floatX), name='delta_b')
        self.delta_b_prime = theano.shared(value=numpy.zeros_like(bvis.get_value(borrow=True), dtype=theano.config.floatX), name='delta_b_prime')

        self.params = [self.W, self.b, self.b_prime]
        self.delta_params = [self.delta_W, self.delta_b, self.delta_b_prime]

    def get_corrupted_input(self, input, corruption_level):
        return  self.theano_rng.binomial(size=input.shape, n=1,
                                         p=1 - corruption_level,
                                         dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return self.hidden_activation(T.dot(input, self.W) + self.b)
    def get_jacobian(self, hidden, W):
        """Computes the jacobian of the hidden layer with respect to
        the input, reshapes are necessary for broadcasting the
        element-wise product on the right axis

        """
        return T.reshape(hidden * (1 - hidden),(self.n_batchsize, 1, self.n_hidden)) * T.reshape(W, (1, self.n_visible, self.n_hidden))
    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the hidden layer """
        return  self.reconstruct_activation(T.dot(hidden, self.W_prime) + self.b_prime)

    def kl_divergence(self, p, p_hat):
        return p * T.log(p / p_hat) + (1 - p) * T.log((1 - p) / (1 - p_hat))

    def get_cost_updates(self, corruption_level, learning_rate, momentum, weight_y = None, size_of_x = None, size_of_y = None):
        """ This function computes the cost and the updates for one training step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        
        
#        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        if weight_y is None :
            L = - T.mean(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        elif weight_y == 0:
            weight_y = float(size_of_y) / size_of_x
            L_constant = T.mean(- T.mean(self.x[:,:size_of_x] * T.log(z[:,:size_of_x]) + (1 - self.x[:,:size_of_x]) * T.log(1 - z[:,:size_of_x]), axis=1)) + \
                T.mean(weight_y * (- T.mean(self.x[:,size_of_x:] * T.log(z[:,size_of_x:]) + (1 - self.x[:,size_of_x:]) * T.log(1 - z[:,size_of_x:]), axis=1)))
            ### this gives a single value of L not a vector
        else:
 #           L = - T.mean(self.x[:,:size_of_x] * T.log(z[:,:size_of_x]) + (1 - self.x[:,:size_of_x]) * T.log(1 - z[:,:size_of_x]), axis=1) + \
 #               weight_y * (- T.mean(self.x[:,size_of_x:] * T.log(z[:,size_of_x:]) + (1 - self.x[:,size_of_x:]) * T.log(1 - z[:,size_of_x:]), axis=1))   
            L_constant = T.mean(- T.mean(self.x[:,:size_of_x] * T.log(z[:,:size_of_x]) + (1 - self.x[:,:size_of_x]) * T.log(1 - z[:,:size_of_x]), axis=1)) + \
                T.mean(weight_y * (- T.mean(self.x[:,size_of_x:] * T.log(z[:,size_of_x:]) + (1 - self.x[:,size_of_x:]) * T.log(1 - z[:,size_of_x:]), axis=1)))     
        if self.reconstruct_activation is T.tanh:
            #L = T.sqr(self.x - z).sum(axis=1)
                if weight_y is None:
                    L = T.sqr(self.x - z).mean(axis=1)
                elif weight_y == 0:
                    weight_y = size_of_x / float(size_of_y)
                    L = T.sqr(self.x[:,:size_of_x] - z[:,size_of_x]).mean(axis=1) + T.sqr(self.x[:,size_of_x:] - z[:,size_of_x:]).mean(axis=1)
                else:
                    L = T.sqr(self.x[:,:size_of_x] - z[:,size_of_x]).mean(axis=1) + T.sqr(self.x[:,size_of_x:] - z[:,size_of_x:]).mean(axis=1)    
               
                
        # Compute the jacobian and average over the number of samples/minibatch
#        J = self.get_jacobian(y, self.W)
#        self.L_jacob = T.sum(J ** 2) / self.n_batchsize
#   add this line to cast cost to datatype of system.
#        self.L_jacob = T.cast(self.L_jacob, dtype = theano.config.floatX) 
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        if self.contraction_level == 0:
                if self.sparsity_weight != 0 :
                    sparsity_level = T.extra_ops.repeat(self.sparsity, self.n_hidden)
                    avg_act = y.mean(axis=0)

                    kl_div = self.kl_divergence(sparsity_level, avg_act)
                    if weight_y is None:
                        cost = T.mean(L) + self.sparsity_weight * kl_div.sum() 
                    else:
                        cost = L_constant + self.sparsity_weight * kl_div.sum() 
                else: # contraction_level is not zero
                    if weight_y is None:
                        cost = T.mean(L)
                    else:
                        cost = L_constant                                                   
        else:
                # Compute the jacobian and average over the number of samples/minibatch
                J = self.get_jacobian(y, self.W)
                self.L_jacob = T.cast((T.sum(J ** 2) / self.n_batchsize), dtype = theano.config.floatX)            
                if self.sparsity_weight is not None:
                    sparsity_level = T.extra_ops.repeat(self.sparsity, self.n_hidden)
                    avg_act = y.mean(axis=0)

                    kl_div = self.kl_divergence(sparsity_level, avg_act)
                    if weight_y is None:
                        cost = T.mean(L) + self.sparsity_weight * kl_div.sum() + self.contraction_level * T.mean(self.L_jacob)
                    else:
                        cost = L_constant + self.sparsity_weight * kl_div.sum() + self.contraction_level * T.mean(self.L_jacob)
                else:
                    if weight_y is None:
                        cost = T.mean(L) + self.contraction_level * T.mean(self.L_jacob)
                    else:
                        cost = L_constant + self.contraction_level * T.mean(self.L_jacob)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters

        cost = T.cast(cost, dtype = theano.config.floatX)
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        
        updates = collections.OrderedDict()
        for dparam, gparam in zip(self.delta_params, gparams):
            updates[dparam] = T.cast(momentum * dparam - gparam*learning_rate, dtype = theano.config.floatX)
        for dparam, param in zip(self.delta_params, self.params):
            updates[param] = T.cast(param + updates[dparam], dtype = theano.config.floatX)
        
#        updates = []
#        for param, gparam in zip(self.params, gparams):
#            updates.append((param, param - learning_rate * gparam))
        return (cost, updates)
    def transform(self, data_x):  # get the last layaer activations to transform data.
        last_layer_activations = self.get_hidden_values(self.x)
        theano_fn = theano.function(inputs=[],
                                 outputs=last_layer_activations,

                                 givens={self.x: data_x})
        newFeatures = theano_fn()
        return newFeatures

class dA_maxout(dA):
    """Denoising Auto-Encoder with Maxout hidden activation"""

    def __init__(self, numpy_rng, theano_rng=None, input=None,
                 n_visible=500, n_hidden=500,
                 W=None, bhid=None, bvis=None, sparsity = None,
                 sparsity_weight = None,
                 reconstruct_activation = (lambda x: 1.0*x),
                 pool_size = 3):
        super(dA_maxout, self).__init__(numpy_rng=numpy_rng, theano_rng=theano_rng,
                 input=input, n_visible=n_visible, n_hidden = n_hidden,
                 W= W, bhid = bhid, bvis = bvis, sparsity = sparsity, 
                 sparsity_weight = sparsity_weight,
                 hidden_activation = (lambda x: 1.0*x),
                 reconstruct_activation = reconstruct_activation)

        self.pool_size = pool_size
        initial_W_prime = numpy.asarray(numpy_rng.uniform(
                  low=-numpy.sqrt(6. / (n_hidden/pool_size + n_visible)),
                  high=numpy.sqrt(6. / (n_hidden/pool_size + n_visible)),
                  size=(n_hidden/pool_size, n_visible)), dtype=theano.config.floatX)
        W_prime = theano.shared(value=initial_W_prime, name='W_prime', borrow=True)

        # tied weights, therefore W_prime is W transpose
        self.W_prime = W_prime
        self.delta_W_prime = theano.shared(value=numpy.zeros_like(W_prime.get_value(borrow=True), dtype=theano.config.floatX),
                                                                  name='delta_W_prime')

        self.params = [self.W, self.W_prime, self.b, self.b_prime]
        self.delta_params = [self.delta_W, self.delta_W_prime, self.delta_bvis, self.delta_bhid]        

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        lin_output = T.dot(input, self.W) + self.b
        last_start = self.n_hidden - self.pool_size
        tmp_output = lin_output[:,0:last_start+1:self.pool_size]
        for i in range(1, self.pool_size):
            cur = lin_output[:,i:last_start+i+1:self.pool_size]
            tmp_output = T.maximum(cur, tmp_output)

        return tmp_output

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the hidden layer """
        if self.reconstruct_activation is None:
            return  T.dot(hidden, self.W_prime) + self.b_prime
        else:
            return  self.reconstruct_activation(T.dot(hidden, self.W_prime) + self.b_prime)


