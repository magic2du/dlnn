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

import theano
from theano.tensor.shared_randomstreams import RandomStreams
from dlnn.io_func.data_io import read_data_args, read_dataset
from dlnn.utils.learn_rates import LearningRateExpDecay
import numpy
import theano.tensor as T
from dlnn.utils.utils import parse_lrate, parse_activation, parse_conv_spec, activation_to_txt, string_2_bool


class NetworkConfig():

    def __init__(self,settings = None):
        self.settings = settings
        self.model_type = 'DNN'

        self.batch_size = 256
        self.momentum = 0.5
        self.lrate = LearningRateExpDecay(start_rate=0.08, scale_by = 0.5,
                                          min_derror_decay_start = 0.05,
                                          min_derror_stop = 0.05,
                                          min_epoch_decay_start=15)

        self.activation = T.nnet.sigmoid
        self.activation_text = 'sigmoid'
        self.do_maxout = False
        self.pool_size = 1
        self.contraction_level = 0

        self.do_dropout = False
        self.dropout_factor = []
        self.input_dropout_factor = 0.0

        self.max_col_norm = None
        self.l1_reg = None
        self.l2_reg = None

        # data reading
        self.train_sets = None
        self.train_xy = None
        self.train_x = None
        self.train_y = None

        self.valid_sets = None
        self.valid_xy = None
        self.valid_x = None
        self.valid_y = None

        self.test_sets = None
        self.test_xy = None
        self.test_x = None
        self.test_y = None 
    

        # specifically for DNN
        self.n_ins = settings['n_ins'] if settings['n_ins'] is not None else 0
        self.hidden_layers_sizes =  settings['hidden_layers_sizes'] if settings['hidden_layers_sizes'] is not None else []
        
#        self.ptr-layer-number = len(settings['hidden_layers_sizes'])
        self.n_outs = settings['n_outs'] if settings['n_outs'] is not None else 0
        self.wdir  = '.'
        self.non_updated_layers = []
        self.numpy_rng = numpy.random.RandomState(89677)
        self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))
        # specifically for DNN_SAT
        self.ivec_n_ins = 0
        self.ivec_hidden_layers_sizes = []
        self.ivec_n_outs = 0 

        # specifically for convolutional networks
        self.conv_layer_configs = []
        self.conv_activation = T.nnet.sigmoid
        self.conv_activation_text = 'sigmoid'
        self.use_fast = False 

        # number of epochs between model saving (for later model resuming)
        self.model_save_step = 1

        # the path to save model into Kaldi-compatible format
        self.cfg_output_file = ''
        self.param_output_file = ''
        self.kaldi_output_file = ''
        
        
        # default setup for Sda
        # parameters related with training 
        self.epochs = 1500                  # number of training epochs for each layer
        self.batch_size = 100            # size of mini-batches
        self.corruption_levels=[0 for n in xrange(100)]  # denoising factor; we use an array for future extension to layer-specific factor
        self.learning_rates = [1 for n in xrange(100)]   # learning rate for each layer
        self.momentum = 0.5              # momentum 

        self.ptr_layer_number = len(self.hidden_layers_sizes)        # number of layers to be trained
        self.hidden_activation = T.nnet.sigmoid   # activation function of the hidden layer/output
        self.firstlayer_reconstruct_activation = T.nnet.sigmoid      # the reconstruction activation function for the first layer
                                                                   # if we normaze the input data with mean (maybe also with variance)
                                                                   # normalization, then we need the tanh activation function to reconstruct
                                                                   # the input
        
        # for maxout autoencoder
        self.do_maxout = False           # whether to apply maxout on the hidden layer
        self.pool_size = 1               # pooling size of maxout

        # interfaces for the training data
        self.train_sets = None
        self.train_xy = None
        self.train_x = None
        self.train_y = None

        # interfaces for validation data. we don't do validation for RBM, so these variables will be None 
        # we have these variables because we want to use the _cfg2file function from io_func/model_io.py
        self.valid_sets = None
        self.valid_xy = None
        self.valid_x = None
        self.valid_y = None



        # parse sparsity settings for sparsity autoencoder
        self.sparsity = None
        self.sparsity_weight = None

        # path to save model
        self.cfg_output_file = ''       # where to save this config class
        self.param_output_file = ''     # where to save the network parameters
        self.kaldi_output_file = ''     # where to save the Kaldi-formatted model
        
        # initialize
        self.parse_config_common(settings)
    # initialize pfile reading. TODO: inteference *directly* for Kaldi feature and alignment files
    def init_data_reading(self, train_data_spec, valid_data_spec = None):
        train_dataset, train_dataset_args = read_data_args(train_data_spec)
        if valid_data_spec is not None:
            valid_dataset, valid_dataset_args = read_data_args(valid_data_spec)
            self.valid_sets, self.valid_xy, self.valid_x, self.valid_y = read_dataset(valid_dataset, valid_dataset_args)
        self.train_sets, self.train_xy, self.train_x, self.train_y = read_dataset(train_dataset, train_dataset_args)

    def init_data_reading_test(self, data_spec):
        dataset, dataset_args = read_data_args(data_spec)
        self.test_sets, self.test_xy, self.test_x, self.test_y = read_dataset(dataset, dataset_args)
    def parse_config_common_sda(self, arguments):
        if arguments.has_key('corruption_level'):
            self.corruption_levels = [float(arguments['corruption_level']) for n in xrange(100)]
        if arguments.has_key('learning_rate'):
            self.learning_rates = [float(arguments['learning_rate']) for n in xrange(100)]
        if arguments.has_key('batch_size'):
            self.batch_size = int(arguments['batch_size'])
        if arguments.has_key('epoch_number'):
            self.epochs = int(arguments['epoch_number'])
        if arguments.has_key('momentum'):
            self.momentum = float(arguments['momentum'])

        # parse DNN network structure
        if arguments.has_key('nnet_spec'):
                nnet_layers = arguments['nnet_spec'].split(':')
                self.n_ins = int(nnet_layers[0])
                self.hidden_layers_sizes = [int(nnet_layers[i]) for i in range(1, len(nnet_layers)-1)]
                self.n_outs = int(nnet_layers[-1])

        # parse pre-training layer number
        self.ptr_layer_number = len(self.hidden_layers_sizes)
        if arguments.has_key('ptr_layer_number'):
            self.ptr_layer_number = int(arguments['ptr_layer_number'])

        # parse activation function
        if arguments.has_key('hidden_activation'):
            self.hidden_activation = parse_activation(arguments['hidden_activation'])
            if arguments['hidden_activation'].startswith('maxout'):
                self.do_maxout = True; self.pool_size = int(arguments['hidden_activation'].replace('maxout:',''))
        if arguments.has_key('1stlayer_reconstruct_activation'):
            self.firstlayer_reconstruct_activation = parse_activation(arguments['1stlayer_reconstruct_activation'])

        # parse sparsity setting
        if arguments.has_key('sparsity'):
            self.sparsity = float(arguments['sparsity'])
        if arguments.has_key('sparsity_weight'):
            self.sparsity_weight = float(arguments['sparsity_weight'])

        # parse various paths for model saving
        if arguments.has_key('cfg_output_file'):
            self.cfg_output_file = arguments['cfg_output_file']
        if arguments.has_key('param_output_file'):
            self.param_output_file = arguments['param_output_file']
        if arguments.has_key('kaldi_output_file'):
            self.kaldi_output_file = arguments['kaldi_output_file']
    def parse_config_common(self, arguments):
        # parse batch_size, momentum and learning rate 
        self.parse_config_common_sda(arguments)
        if arguments.has_key('batch_size'):
            self.batch_size = int(arguments['batch_size'])
        if arguments.has_key('momentum'):
            self.momentum = float(arguments['momentum'])
        if arguments.has_key('lrate'):
            self.lrate = parse_lrate(arguments['lrate'])
        if arguments.has_key('wdir'):
            self.wdir  = arguments['wdir']
        if arguments.has_key('contraction_level'):
            self.contraction_level  = arguments['contraction_level']
        # parse activation function, including maxout
        if arguments.has_key('activation'):
            self.activation_text = arguments['activation']
            self.activation = parse_activation(arguments['activation'])
            if arguments['activation'].startswith('maxout'):
                self.do_maxout = True
                self.pool_size = int(arguments['activation'].replace('maxout:',''))
                self.activation_text = 'maxout'

        # parse dropout. note that dropout can be applied to the input features only when dropout is also
        # applied to hidden-layer outputs at the same time. that is, you cannot apply dropout only to the
        # input features
        if arguments.has_key('dropout_factor'):
            self.do_dropout = True
            self.dropout_factor = [float(arguments['dropout_factor']) for n in xrange(20)]
        if arguments.has_key('input_dropout_factor'):
            self.input_dropout_factor = float(arguments['input_dropout_factor'])
        if arguments.has_key('l2_reg'):
            self.l2_reg = arguments['l2_reg'] 
        if arguments.has_key('l1_reg'):
            self.l1_reg = arguments['l1_reg'] 
        if arguments.has_key('cfg_output_file'):
            self.cfg_output_file = arguments['cfg_output_file'] 
        if arguments.has_key('param_output_file'):
            self.param_output_file = arguments['param_output_file']
        if arguments.has_key('kaldi_output_file'):
            self.kaldi_output_file = arguments['kaldi_output_file']

        if arguments.has_key('model_save_step'):
            self.model_save_step = int(arguments['model_save_step'])

        if arguments.has_key('non_updated_layers'):
            layers = arguments['non_updated_layers'].split(",")
            self.non_updated_layers = [int(layer) for layer in layers]

    def parse_config_dnn(self, arguments, nnet_spec):
        self.parse_config_common(arguments)
        # parse DNN network structure
        nnet_layers = nnet_spec.split(':')
        self.n_ins = int(nnet_layers[0])
        self.hidden_layers_sizes = [int(nnet_layers[i]) for i in range(1, len(nnet_layers)-1)]
        self.n_outs = int(nnet_layers[-1])

    def parse_config_cnn(self, arguments, nnet_spec, conv_nnet_spec):
        self.parse_config_dnn(arguments, nnet_spec)
        # parse convolutional layer structure
        self.conv_layer_configs = parse_conv_spec(conv_nnet_spec, self.batch_size)
        # parse convolutional layer activation
        # parse activation function, including maxout
        if arguments.has_key('conv_activation'):
            self.conv_activation_text = arguments['conv_activation']
            self.conv_activation = parse_activation(arguments['conv_activation'])
            # maxout not supported yet
        # whether we use the fast version of convolution 
        if arguments.has_key('use_fast'):
            self.use_fast = string_2_bool(arguments['use_fast'])

