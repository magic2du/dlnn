from DL_libs import *


train_data_spec = arguments['train_data']
valid_data_spec = arguments['valid_data']
conv_nnet_spec = arguments['conv_nnet_spec']
nnet_spec = arguments['nnet_spec']
wdir = arguments['wdir']

# parse network configuration from arguments, and initialize data reading
cfg = NetworkConfig(); cfg.model_type = 'CNN'
cfg.parse_config_cnn(arguments, '10:' + nnet_spec, conv_nnet_spec)
cfg.init_data_reading(train_data_spec, valid_data_spec)

# parse pre-training options
# pre-training files and layer number (how many layers are set to the pre-training parameters)
ptr_layer_number = 0; ptr_file = ''
if arguments.has_key('ptr_file') and arguments.has_key('ptr_layer_number'):
    ptr_file = arguments['ptr_file']
    ptr_layer_number = int(arguments['ptr_layer_number'])

# check working dir to see whether it's resuming training
resume_training = False
if os.path.exists(wdir + '/nnet.tmp') and os.path.exists(wdir + '/training_state.tmp'):
    resume_training = True
    cfg.lrate = _file2lrate(wdir + '/training_state.tmp')
    log('> ... found nnet.tmp and training_state.tmp, now resume training from epoch ' + str(cfg.lrate.epoch))

numpy_rng = numpy.random.RandomState(89677)
theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
log('> ... initializing the model')
# construct the cnn architecture
cnn = CNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg)
# load the pre-training networks, if any, for parameter initialization
if (ptr_layer_number > 0) and (resume_training is False):
    _file2nnet(cnn.layers, set_layer_num = ptr_layer_number, filename = ptr_file)
if resume_training:
    _file2nnet(cnn.layers, filename = wdir + '/nnet.tmp')