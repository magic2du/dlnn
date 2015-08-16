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

from dlnn.io_func.model_io import _nnet2file, _file2nnet
from dlnn.layers.logistic_sgd import LogisticRegression
from dlnn.layers.mlp import HiddenLayer, DropoutHiddenLayer, _dropout_from_layer
import theano.tensor as T
from dlnn.io_func.model_io import _nnet2file, _cfg2file, _file2nnet, log
from dlnn.utils.utils import shared_dataset_X, shared_dataset
from dlnn.learning.sgd import train_sgd, validate_by_minibatch, validate_by_minibatch_without_streaming, train_sgd_without_streaming

class DNN(object):

    def __init__(self, numpy_rng, theano_rng=None,
                 cfg = None,  # the network configuration
                 dnn_shared = None, shared_layers=[], input = None):

        self.layers = []
        self.params = []
        self.delta_params = []

        self.cfg = cfg
        self.n_ins = cfg.n_ins; self.n_outs = cfg.n_outs if cfg.n_outs !=1 else 2
        self.hidden_layers_sizes = cfg.hidden_layers_sizes
        self.hidden_layers_number = len(self.hidden_layers_sizes)
        self.activation = cfg.activation

        self.do_maxout = cfg.do_maxout; self.pool_size = cfg.pool_size

        self.max_col_norm = cfg.max_col_norm
        self.l1_reg = cfg.l1_reg
        self.l2_reg = cfg.l2_reg

        self.non_updated_layers = cfg.non_updated_layers

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        if input == None:
            self.x = T.matrix('x')
        else:
            self.x = input 
        self.y = T.ivector('y')

        for i in xrange(self.hidden_layers_number):
            # construct the hidden layer
            if i == 0:
                input_size = self.n_ins
                layer_input = self.x
            else:
                input_size = self.hidden_layers_sizes[i - 1]
                layer_input = self.layers[-1].output

            W = None; b = None
            if (i in shared_layers) :
                W = dnn_shared.layers[i].W; b = dnn_shared.layers[i].b
            if self.do_maxout == True:
                hidden_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=self.hidden_layers_sizes[i] * self.pool_size,
                                        W = W, b = b,
                                        activation = (lambda x: 1.0*x),
                                        do_maxout = True, pool_size = self.pool_size)
            else:
                hidden_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=self.hidden_layers_sizes[i],
                                        W = W, b = b,
                                        activation=self.activation)
            # add the layer to our list of layers
            self.layers.append(hidden_layer)
            # if the layer index is included in self.non_updated_layers, parameters of this layer will not be updated
            if (i not in self.non_updated_layers):
                self.params.extend(hidden_layer.params)
                self.delta_params.extend(hidden_layer.delta_params)
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
                         input=self.layers[-1].output,
                         n_in=self.hidden_layers_sizes[-1], n_out=self.n_outs)

        if self.n_outs > 0:
            self.layers.append(self.logLayer)
            self.params.extend(self.logLayer.params)
            self.delta_params.extend(self.logLayer.delta_params)
       
        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)

        if self.l1_reg is not None:
            for i in xrange(self.hidden_layers_number):
                W = self.layers[i].W
                self.finetune_cost += self.l1_reg * (abs(W).sum())

        if self.l2_reg is not None:
            for i in xrange(self.hidden_layers_number):
                W = self.layers[i].W
                self.finetune_cost += self.l2_reg * T.sqr(W).sum()
    def predict(self, x_dataset):
        predict_fn = theano.function([], self.logLayer.y_pred,
                   givens={self.x: x_dataset})
        predicted = predict_fn()
        return predicted
    def predict_p(self, x_dataset):
        predict_p_fn = theano.function([], self.logLayer.p_y_given_x,
                   givens={self.x: x_dataset})
        predicted_p = predict_p_fn()
        return predicted_p

    def build_finetune_functions(self, train_shared_xy, valid_shared_xy, batch_size):

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')
        momentum = T.fscalar('momentum')

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = collections.OrderedDict()
        for dparam, gparam in zip(self.delta_params, gparams):
            updates[dparam] = momentum * dparam - gparam*learning_rate
        for dparam, param in zip(self.delta_params, self.params):
            updates[param] = param + updates[dparam]

        if self.max_col_norm is not None:
            for i in xrange(self.hidden_layers_number):
                W = self.layers[i].W
                if W in updates:
                    updated_W = updates[W]
                    col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                    desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                    updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

        train_fn = theano.function(inputs=[index, theano.Param(learning_rate, default = 0.0001),
              theano.Param(momentum, default = 0.5)],
              outputs=self.errors,
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})

        valid_fn = theano.function(inputs=[index],
              outputs=self.errors,
              givens={
                self.x: valid_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size:
                                    (index + 1) * batch_size]})

        return train_fn, valid_fn


    def build_extract_feat_function(self, output_layer):

        feat = T.matrix('feat')
        out_da = theano.function([feat], self.layers[output_layer].output, updates = None, givens={self.x:feat}, on_unused_input='warn')
        return out_da
    def transform(self, x):
        fn = self.build_extract_feat_function(self.hidden_layers_number -1)
        new_x = fn(x)
        return new_x
    def build_finetune_functions_kaldi(self, train_shared_xy, valid_shared_xy):

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')
        momentum = T.fscalar('momentum')

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = collections.OrderedDict()
        for dparam, gparam in zip(self.delta_params, gparams):
            updates[dparam] = momentum * dparam - gparam*learning_rate
        for dparam, param in zip(self.delta_params, self.params):
            updates[param] = param + updates[dparam]

        if self.max_col_norm is not None:
            for i in xrange(self.hidden_layers_number):
                W = self.layers[i].W
                if W in updates:
                    updated_W = updates[W]
                    col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                    desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                    updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

        train_fn = theano.function(inputs=[theano.Param(learning_rate, default = 0.0001),
              theano.Param(momentum, default = 0.5)],
              outputs=self.errors,
              updates=updates,
              givens={self.x: train_set_x, self.y: train_set_y})

        valid_fn = theano.function(inputs=[],
              outputs=self.errors,
              givens={self.x: valid_set_x, self.y: valid_set_y})

        return train_fn, valid_fn

    def write_model_to_raw(self, file_path):
        # output the model to tmp_path; this format is readable by PDNN
        _nnet2file(self.layers, filename=file_path)

    def write_model_to_kaldi(self, file_path, with_softmax = True):
        # determine whether it's BNF based on layer sizes
        output_layer_number = -1;
        for layer_index in range(1, self.hidden_layers_number - 1):
            cur_layer_size = self.hidden_layers_sizes[layer_index]
            prev_layer_size = self.hidden_layers_sizes[layer_index-1]
            next_layer_size = self.hidden_layers_sizes[layer_index+1]
            if cur_layer_size < prev_layer_size and cur_layer_size < next_layer_size:
                output_layer_number = layer_index+1; break

        layer_number = len(self.layers)
        if output_layer_number == -1:
            output_layer_number = layer_number

        fout = open(file_path, 'wb')
        for i in xrange(output_layer_number):
            activation_text = '<' + self.cfg.activation_text + '>'
            if i == (layer_number-1) and with_softmax:   # we assume that the last layer is a softmax layer
                activation_text = '<softmax>'
            W_mat = self.layers[i].W.get_value()
            b_vec = self.layers[i].b.get_value()
            input_size, output_size = W_mat.shape
            W_layer = []; b_layer = ''
            for rowX in xrange(output_size):
                W_layer.append('')

            for x in xrange(input_size):
                for t in xrange(output_size):
                    W_layer[t] = W_layer[t] + str(W_mat[x][t]) + ' '

            for x in xrange(output_size):
                b_layer = b_layer + str(b_vec[x]) + ' '

            fout.write('<affinetransform> ' + str(output_size) + ' ' + str(input_size) + '\n')
            fout.write('[' + '\n')
            for x in xrange(output_size):
                fout.write(W_layer[x].strip() + '\n')
            fout.write(']' + '\n')
            fout.write('[ ' + b_layer.strip() + ' ]' + '\n')
            if activation_text == '<maxout>':
                fout.write(activation_text + ' ' + str(output_size/self.pool_size) + ' ' + str(output_size) + '\n')
            else:
                fout.write(activation_text + ' ' + str(output_size) + ' ' + str(output_size) + '\n')
        fout.close()
    def finetuning(self, train_xy, valid_xy):
            # get the training, validation and testing function for the model
            log('> ... getting the finetuning functions')
            train_shared_xy = shared_dataset(train_xy, borrow=True)
            valid_shared_xy = shared_dataset(valid_xy, borrow=True)
            train_fn, valid_fn = self.build_finetune_functions(
                        train_shared_xy = train_shared_xy, valid_shared_xy = valid_shared_xy,
                        batch_size = self.cfg.batch_size)

            log('> ... finetuning the model')
            while (self.cfg.lrate.get_rate() != 0):
                # one epoch of sgd training 
                train_error = train_sgd_without_streaming(train_fn, train_xy[0].shape[0], self.cfg)
                log('> epoch %d, training error %f ' % (self.cfg.lrate.epoch, 100*numpy.mean(train_error)) + '(%)')
                # validation 
                valid_error = validate_by_minibatch_without_streaming(valid_fn, valid_xy[0].shape[0], self.cfg)
                log('> epoch %d, lrate %f, validation error %f ' % (self.cfg.lrate.epoch, self.cfg.lrate.get_rate(), 100*numpy.mean(valid_error)) + '(%)')
                self.cfg.lrate.get_next_rate(current_error = 100*numpy.mean(valid_error))
            log('> ... finetuning finished')
                # output nnet parameters and lrate, for training resume
#                if self.cfg.lrate.epoch % self.cfg.model_save_step == 0:
#                    _nnet2file(dnn.layers, filename=wdir + '/nnet.tmp')
 #                   _lrate2file(cfg.lrate, wdir + '/training_state.tmp') 
            '''
            # save the model and network configuration
            if self.cfg.param_output_file != '':
                _nnet2file(self.layers, filename=self.cfg.param_output_file, input_factor = self.cfg.input_dropout_factor, factor = self.cfg.dropout_factor)
                log('> ... the final PDNN model parameter is ' + self.cfg.param_output_file)
            if self.cfg.cfg_output_file != '':
                _cfg2file(self.cfg, filename=self.cfg.cfg_output_file)
                log('> ... the final PDNN model config is ' + self.cfg.cfg_output_file)

            # output the model into Kaldi-compatible format
            if self.cfg.kaldi_output_file != '':
                self.write_model_to_kaldi(self.cfg.kaldi_output_file)
                log('> ... the final Kaldi model is ' + self.cfg.kaldi_output_file) 

            # remove the tmp files (which have been generated from resuming training) 
#            os.remove(wdir + '/nnet.tmp')
#            os.remove(wdir + '/training_state.tmp')         
        
            '''
    def load_pretrain_from_Sda(self, sda_xy = None):
            
            for i in xrange(len(self.cfg.hidden_layers_sizes)):
                layer = self.layers[i]
                sda_xy_layer = sda_xy.dA_layers[i]
                if layer.type == 'fc':
                    W_xy = sda_xy_layer.W.get_value()
                    W_dnn = layer.W.get_value()
                    x_size, y_size = W_dnn.shape
                    b_size = layer.b.get_value().shape[0]
                    layer.W.set_value(W_xy[ :x_size, :y_size])
                    layer.b.set_value(sda_xy_layer.b.get_value()[:b_size])
                elif layer.type == 'conv':
                    pass