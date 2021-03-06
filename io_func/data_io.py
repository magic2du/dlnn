# Copyright 2013    Yajie Miao    Carnegie Mellon University 

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

import glob
import gzip
import numpy
import os
import sys, re
import theano

from dlnn.io_func.kaldi_io import KaldiDataRead
from dlnn.io_func.pfile_io import PfileDataRead, PfileDataReadStream
from dlnn.io_func.pickle_io import PickleDataRead
import theano.tensor as T
from dlnn.utils.utils import string_2_bool


def read_data_args(data_spec):
    elements = data_spec.split(",")
    pfile_path_list = glob.glob(elements[0])
    dataset_args = {}
    # default settings
    dataset_args['type'] = 'pickle'
    dataset_args['random'] = False
    dataset_args['stream'] = False
    dataset_args['partition'] = 1024 * 1024 * 600  # by default the partition size is 600m if stream is True

    # the type of the data: pickle, pfile   TO-DO: HDF5
    if '.pickle' in data_spec:
        dataset_args['type'] = 'pickle'
    elif '.pfile' in data_spec:
        dataset_args['type'] = 'pfile'
    elif '.scp' in data_spec:
        dataset_args['type'] = 'kaldi'
    else:
        dataset_args['type'] = ''

    for i in range(1, len(elements)):
        element = elements[i]
        arg_value = element.split("=")
        value = arg_value[1]
        key = arg_value[0]
        if key == 'partition':
            dataset_args['partition'] = 1024 * 1024 * int(value.replace('m',''))
        elif key == 'stream':
            dataset_args['stream'] = string_2_bool(value) # not supported for now
        elif key == 'random':
            dataset_args['random'] = string_2_bool(value)
        elif key == 'label':
            dataset_args['label'] = value
        else:
            dataset_args[key] = value  # left context & right context; maybe different
    return pfile_path_list, dataset_args

def read_dataset(file_path_list, read_opts):
    if read_opts['type'] == 'pickle':
        data_reader = PickleDataRead(file_path_list, read_opts)
    elif read_opts['type'] == 'pfile':
        if read_opts['stream']:
            data_reader = PfileDataReadStream(file_path_list, read_opts)
        else:
            data_reader = PfileDataRead(file_path_list, read_opts)
    elif read_opts['type'] == 'kaldi':
        data_reader = KaldiDataRead(file_path_list, read_opts)

    data_reader.initialize_read(first_time_reading = True)
    
    shared_xy = data_reader.make_shared()
    shared_x, shared_y = shared_xy
    shared_y = T.cast(shared_y, 'int32')

    return data_reader, shared_xy, shared_x, shared_y
