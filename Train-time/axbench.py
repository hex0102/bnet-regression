
from __future__ import print_function

import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import binary_net

from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial

from collections import OrderedDict



def min_max(x):
  
  Min=x.min(axis=0)
  Max=x.max(axis=0)
  
  diff=Max-Min
  #diff=max(x)-min(x) 
  diff=np.asarray([i if(i!=0) else 1 for i in diff])
  x=np.asarray((x-Min)/diff,dtype='float32')
  return x

if __name__ == "__main__":
    
    # BN parameters
    batch_size = 50
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    # alpha = .15
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-5
    print("epsilon = "+str(epsilon))
    
    # MLP parameters
    num_units = 256
    print("num_units = "+str(num_units))
    n_hidden_layers = 2
    print("n_hidden_layers = "+str(n_hidden_layers))
    
    # Training parameters
    num_epochs = 1000
    print("num_epochs = "+str(num_epochs))
    
    # Dropout parameters
    dropout_in = .2 # 0. means no dropout
    print("dropout_in = "+str(dropout_in))
    dropout_hidden = .5
    print("dropout_hidden = "+str(dropout_hidden))
    
    # BinaryOut
    activation = binary_net.binary_tanh_unit
    print("activation = binary_net.binary_tanh_unit")
    # activation = binary_net.binary_sigmoid_unit
    # print("activation = binary_net.binary_sigmoid_unit")
    
    # BinaryConnect
    binary = True
    print("binary = "+str(binary))
    stochastic = False
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.    
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))
    
    # Decaying LR 
    LR_start = .005
    print("LR_start = "+str(LR_start))
    LR_fin = 0.0000005
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    save_path = "axbench_parameters.npz"
    print("save_path = "+str(save_path))
    
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))
    
    print('Loading axbench dataset...')
   
    
    with open("train.txt",'r') as f:
      trainlines = f.readlines()

    infoline = trainlines[0].split(' ')
    totalsize = infoline[0]
   
    infile = trainlines[1:]
    input_set = []
    output_set = []
   
    print(len(infile)/2)
 
    for i in xrange(len(infile)/2):
      perinput = infile[2*i].split(' ')[1:]
      input_set.append([float(x) for x in perinput])
      peroutput = infile[2*i+1].split(' ')[1:]
      output_set.append([float(x) for x in peroutput])

    totalsample = len(output_set)
    # print(totalsize )
    # print(totalsample )
    # assert totalsize == totalsample
    input_set=np.asarray(input_set)
    output_set=np.asarray(output_set)  
 
    train_x = input_set[:80000]
    valid_x = input_set[80000:90000]
    test_x = input_set[90000:100000]

    train_y = output_set[:80000]
    valid_y = output_set[80000:90000]
    test_y = output_set[90000:100000]
    
    train_x = np.asarray(train_x)
    valid_x = np.asarray(valid_x)
    test_x = np.asarray(test_x)     

    train_y = np.asarray(train_y)
    valid_y = np.asarray(valid_y)
    test_y = np.asarray(test_y)

    train_x = min_max(train_x)
    valid_x = min_max(valid_x)
    test_x = min_max(test_x)
    train_y = min_max(train_y)
    valid_y = min_max(valid_y)
    test_y = min_max(test_y)

    train_y = 2 * train_y - 1
    valid_y = 2 * valid_y - 1
    test_y = 2 * test_y - 1
    
    print(len(test_y))
    train_x = 2 * train_x.reshape(-1, 1, 1, 6) - 1
    valid_x = 2 * valid_x.reshape(-1, 1, 1, 6) - 1
    test_x = 2 * test_x.reshape(-1, 1, 1, 6) - 1  
    
    #print(train_x[2])
    print(train_x.ndim) 
    print(">>> Preprocessing finished...") 
    #train_set = MNIST(which_set= 'train', start=0, stop = 50000, center = False)
    #valid_set = MNIST(which_set= 'train', start=50000, stop = 60000, center = False)
    #test_set = MNIST(which_set= 'test', center = False)
    

    print('Building the MLP...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    mlp = lasagne.layers.InputLayer(
            shape= (None, 1, 1, 6), # 28, 28),
            input_var=input)
            
    mlp = lasagne.layers.DropoutLayer(
            mlp, 
            p=0.001)#dropout_in)
    
    for k in range(n_hidden_layers):

        mlp = binary_net.DenseLayer(
                mlp, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_units)                  
        
        mlp = lasagne.layers.BatchNormLayer(
                mlp,
                epsilon=epsilon, 
                alpha=alpha)

        mlp = lasagne.layers.NonlinearityLayer(
                mlp,
                nonlinearity=activation)
                
        mlp = lasagne.layers.DropoutLayer(
                mlp, 
                p=0.001)#dropout_hidden)
    
    mlp = binary_net.DenseLayer(
                mlp, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=1)    
    
    mlp = lasagne.layers.BatchNormLayer(
            mlp,
            epsilon=epsilon, 
            alpha=alpha)

    train_output = lasagne.layers.get_output(mlp, deterministic=False)
    
    # squared hinge loss
    loss = (abs(train_output-target)).mean()+ (lasagne.objectives.squared_error(train_output, target)).mean()
    #T.mean((T.maximum(0.,T.abs_(target-train_output))))
    if binary:
        # W updates
        W = lasagne.layers.get_all_params(mlp, binary=True)
        W_grads = binary_net.compute_grads(loss,mlp)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = binary_net.clipping_scaling(updates,mlp)
        
        # other parameters updates
        params = lasagne.layers.get_all_params(mlp, trainable=True, binary=False)
        updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
        
    else:
        params = lasagne.layers.get_all_params(mlp, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    print(test_output)
    print('>>> stop')
    #test_loss = T.mean((T.maximum(0.,T.abs_(target-test_output))))
    test_loss=(abs(test_output-target)).mean()
    test_err = T.mean(T.abs_(target-test_output) > 0.1*target, dtype = theano.config.floatX)
    #test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err, test_output])

    print('Training...')
    
    binary_net.train(
            train_fn,val_fn,
            mlp,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_x,train_y,
            valid_x,valid_y,
            test_x,test_y,
            save_path,
            shuffle_parts)
