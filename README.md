DLNN
====

DLNN is a lightweight, easy-to-use deep learning toolkit developed under the [Theano](http://deeplearning.net/software/theano) environment. 

Factories and new network models has been added. In the factories, we support arbitrary combinations of network settings, like dropout, sparsity, denoising, contraction, momentum, L1 and L2. The API is simple to use. Stacked parallel auto encoders, and Stacked target-aware autoencoders are supported.

This is an extension to PDNN (http://www.cs.cmu.edu/~ymiao/pdnntk.html). Special thanks to PDNN.
factories -- the factories generate different models, include parallel auto encoders,
cmds     -- commands to conduct general-purpose deep learning  
cmds2    -- additional commands specifically for ASR  
examples -- example setups  
io_func  -- data reading functions; model IO functions; model format conversion  
layers   -- layers: convolution, fully-connected, denoising autoencoder, logistic regression, etc  
learning -- learning algorithms  
models   -- models: CNNs, DNNs, SDAs, SRBMs, etc  
utils    -- utility functions: learning rates, argument parsing, etc
factories -- the factories generate different models, include stacked parallel auto encoders, stacked target-aware autoencoders.

Copyright 2015    Tianchuan Du    University of Delaware
Licensed under the Apache License, Version 2.0 (the "License");

