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
import numpy as np
import os
import sys
import theano
from theano.tensor.shared_randomstreams import RandomStreams
import time

from dlnn.io_func.model_io import _nnet2file, _cfg2file, _file2nnet, log
from dlnn.utils.learn_rates import _lrate2file, _file2lrate
from dlnn.models.dnn import DNN
from dlnn.models.dropout_nnet import DNN_Dropout
from dlnn.models.sda import SdA, Sda_xy
import theano.tensor as T
from dlnn.utils.network_config import NetworkConfig
from dlnn.utils.sda_config import SdAConfig
from dlnn.utils.utils import parse_arguments, save_two_integers, read_two_integers
from dlnn.learning.sgd import train_sgd, validate_by_minibatch
from utils.utils import shared_dataset_X
from numpy import dtype, shape
from DL_libs import cal_epochs 

class Sda_factory:
    def __init__(self, settings=None, data = None):
            """ Stacked Denoising contraction sparse Autoencoders for DNN """
            self.cfg = NetworkConfig(settings)
            # Whether to use dropout            
            self.dnn = DNN_Dropout(numpy_rng=self.cfg.numpy_rng, theano_rng=self.cfg.theano_rng, cfg=self.cfg) if self.cfg.do_dropout else DNN(numpy_rng=self.cfg.numpy_rng, theano_rng=self.cfg.theano_rng, cfg=self.cfg)

            # now set up the SdA model with dnn as an argument
            self.sda = SdA(numpy_rng=self.cfg.numpy_rng, theano_rng=self.cfg.theano_rng, cfg=self.cfg, dnn=self.dnn)

class Sda_xy_factory(Sda_factory):
    def __init__(self, settings=None):
            """ Stacked target-aware auto encoders. Stacked Denoising contraction sparse Autoencoders for pre-training with features and labels.
                The unsupervised pretraining process take Label Y as an a feature and the reconstruction error of Y to the loss function. 
            """
            self.settings = settings
            self.cfg = NetworkConfig(settings)
            self.sda = Sda_xy(numpy_rng=self.cfg.numpy_rng, theano_rng=self.cfg.theano_rng, cfg=self.cfg)

class DNN_factory(Sda_factory):
    def __init__(self, settings=None):
            """ Generates a deep neural network."""
            self.settings = settings
            self.cfg = NetworkConfig(settings)
            self.dnn = DNN_Dropout(numpy_rng=self.cfg.numpy_rng, theano_rng=self.cfg.theano_rng, cfg=self.cfg) if self.cfg.do_dropout else DNN(numpy_rng=self.cfg.numpy_rng, theano_rng=self.cfg.theano_rng, cfg=self.cfg)
            
    # Load the pretrained network from file.       
    def load_pretrain(self, pretrain_file = None, sda = None):
        if (self.cfg.ptr_layer_number > 0) and (pretrain_file is None):
            _file2nnet(self.dnn.layers, set_layer_num = self.cfg.ptr_layer_number, filename = self.cfg.ptr_file)
            
    # Load the pretained network structure from Sda_xy.
    def load_pretrain_from_Sda_xy(self, sda_xy = None):
            
            for i in xrange(len(self.cfg.hidden_layers_sizes)):
                layer = self.dnn.layers[i]
                sda_xy_layer = sda_xy.dA_layers[i]
                if layer.type == 'fc':
                    W_xy = sda_xy_layer.W.get_value()
                    layer.W.set_value(W_xy[ :-self.cfg.n_outs, :])
                    layer.b.set_value(sda_xy_layer.b.get_value())
                elif layer.type == 'conv':
                    filter_shape = layer.filter_shape
                    W_array = layer.W.get_value()
                    for next_X in xrange(filter_shape[0]):
                        for this_X in xrange(filter_shape[1]):
                            new_dict_a = dict_a + ' ' + str(next_X) + ' ' + str(this_X)
                            W_array[next_X, this_X, :, :] = factor * np.asarray(string_2_array(nnet_dict[new_dict_a]), dtype=theano.config.floatX)
                    layer.W.set_value(W_array)

class Parallel_Sda_factory(Sda_factory):
    def __init__(self, settings=None):
            """ Generate a parallel Denoising contraction sparse Autoencoders"""
            self.settings = settings
            self.cfg = NetworkConfig(settings)
    def supervised_training(self, x_train_minmax, x_validation_minmax, y_train_minmax, y_validation_minmax):
            settings = self.settings
            print 'new deep learning using split network'
            # Autoencoder A
            train_x = x_train_minmax[:, :x_train_minmax.shape[1]/2] 
            print "original shape for A", train_x.shape
            cfg = self.settings.copy()
            batch_size = settings['batch_size']
            cfg['epoch_number'] = cal_epochs(settings['pretraining_interations'], x_train_minmax, batch_size = batch_size)   
            cfg['n_ins'] = train_x.shape[1]                    
            sdafA = Sda_factory(cfg)
            self.sdafA = sdafA
            a_MAE_A = sdafA.sda
            a_MAE_A.pretraining(train_x = train_x)                     
            new_x_train_minmax_A =  a_MAE_A.transform(x_train_minmax[:, :x_train_minmax.shape[1]/2])
            self.a_MAE_A = a_MAE_A
            
            # Autoencoder B
            train_x = x_train_minmax[:, x_train_minmax.shape[1]/2:]
            sdafB = Sda_factory(cfg)
            self.sdafB = sdafB                   
            a_MAE_B = sdafB.sda
            a_MAE_B.pretraining(train_x = train_x)                           
            print "original shape for B", train_x.shape
            
            new_x_train_minmax_B =  a_MAE_B.transform(x_train_minmax[:, x_train_minmax.shape[1]/2:])
            self.a_MAE_B = a_MAE_B

            new_x_validation_minmax_A = a_MAE_A.transform(x_validation_minmax[:, :x_validation_minmax.shape[1]/2])
            new_x_validation_minmax_B = a_MAE_B.transform(x_validation_minmax[:, x_validation_minmax.shape[1]/2:])
            new_x_train_minmax_whole = np.hstack((new_x_train_minmax_A, new_x_train_minmax_B))
            
            new_x_validationt_minmax_whole = np.hstack((new_x_validation_minmax_A, new_x_validation_minmax_B))

            # train sda with seperately transformed data
            train_x = new_x_train_minmax_whole
            cfg = settings.copy()
            cfg['epoch_number'] = cal_epochs(settings['pretraining_interations'], new_x_train_minmax_whole, batch_size = batch_size)   
            cfg['n_ins'] = train_x.shape[1]                          
            sdaf = Sda_factory(cfg)                   

            sdaf.sda.pretraining(train_x = train_x)
            sdaf.dnn.finetuning((new_x_train_minmax_whole, y_train_minmax), (new_x_validationt_minmax_whole, y_validation_minmax))
            self.sdaf = sdaf
            self.sda_transformed = sdaf.dnn
    def predict(self, x_test_minmax):
            new_x_test_minmax_A = self.a_MAE_A.transform(x_test_minmax[:, :x_test_minmax.shape[1]/2])
            new_x_test_minmax_B = self.a_MAE_B.transform(x_test_minmax[:, x_test_minmax.shape[1]/2:])
            new_x_test_minmax_whole = np.hstack((new_x_test_minmax_A, new_x_test_minmax_B))
            return self.sda_transformed.predict(new_x_test_minmax_whole)
        
if __name__=='__main__':
        # test the factories:
        settings = {}
        settings['learning_rate'] = 0.01
        settings['train-data'] = "/home/du/Dropbox/Project/libs/dlnn/cmds/train.pickle.gz,partition=600m"  
        settings['hidden_layers_sizes'] = [1024, 1024, 1024]
        settings['n_outs'] = 10 
        settings['batch_size'] = 100
        settings['epoch_number'] = 200
        settings['sparsity']= 0.05 
        settings['sparsity_weight'] =  0.1
        settings['dropout_factor'] = 0.5
        settings['l2_reg'] = 0.05
        settings['l1_reg'] = 0
        settings['weight_y'] = 1 # this is the pretraining penalty of label y.        
        settings['contraction_level'] = 0.01
        settings['momentum'] = 0.2
        settings['wdir'] = '.'
        settings['param-output-file']= 'sda.mdl'
        train_data = cPickle.load(gzip.open('/home/du/Dropbox/Project/libs/dlnn/cmds/train.pickle.gz'))
        valid_data = cPickle.load(gzip.open('/home/du/Dropbox/Project/libs/dlnn/cmds/valid.pickle.gz'))
        train_x, train_y = train_data
        settings['n_ins'] = train_x.shape[1]
        sdaf = Sda_xy_factory(settings)
        sda = sdaf.pretraining_a_sda(train_x, train_y)
        dnnf = DNN_factory(settings)
        dnnf.load_pretrain_from_Sda_xy(sdaf.sda)
        dnnf.dnn.finetuning(train_data, valid_data)
        for dA in sdaf.sda.dA_layers:
            print 'w',dA.W.get_value().shape
            print 'b',dA.b.get_value().shape
        for dA in dnnf.dnn.layers:
            print 'w',dA.W.get_value().shape
            print 'b',dA.b.get_value().shape    
