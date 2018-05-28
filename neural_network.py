#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 12:32:02 2018

@author: Eon
"""

import numpy as np

#%%

from keras.datasets import mnist

#%%

import pickle as pkl

#%%

data = mnist.load_data()


#%%

(trainX,trainY),(testX,testY) = data

#%%
trainX = trainX.reshape((60000,784))/255 - 0.5
testX = testX.reshape((10000,784))/255 - 0.5

#%% 

# onehot

def onehot(T):
    train_y = np.zeros((T.shape[0],10))
    for i,d in enumerate(T):
        train_y[i,d-1] = 1
    return train_y

trainY = onehot(trainY)
testY = onehot(testY)
    
#%%

def _gen_layer_weights(m, n):
    return np.random.normal(0,1,(m,n))

#%%

class MultilayerNN:
    def __init__(self, input_size, output_layer, hidden_layers):
        self.w = []
        if hidden_layers == None or len(hidden_layers) == 0:
            self.w.append(_gen_layer_weights(input_size+1,output_layer))
        layers = len(hidden_layers)
        if layers == 1:
            self.w.append(_gen_layer_weights(input_size+1,hidden_layers[0]))
            self.w.append(_gen_layer_weights(hidden_layers[-1]+1,output_layer))
        else:
            self.w.append(_gen_layer_weights(input_size+1,hidden_layers[0]))
            for i in range(1,layers):
                self.w.append(_gen_layer_weights(hidden_layers[i-1]+1,hidden_layers[i]))
            self.w.append(_gen_layer_weights(hidden_layers[-1]+1,output_layer))
        self.nlayers = len(self.w)
        
#%%
def _sigmoid(self,X, deriv=False):
    if deriv:
        return self._sigmoid(X)*self._sigmoid(1-X)
    return (1/(1+np.exp(-X)))

MultilayerNN._sigmoid = _sigmoid
            
#%%
        
def train(self,X,Y,LR,epochs):
    j = np.zeros((epochs,1))
    m = Y.shape[0]
    for i in range(epochs):
        out = self._train_epoch(X,Y,LR)
        j[i] = self._cost(out,Y)
        acc = np.sum(np.argmax(out,axis=1) == np.argmax(Y,axis=1))/m
        print("%d - Acc: %f Cost: %f"%(i,acc,j[i]))
    return j

MultilayerNN.train = train
        

#%%

def _cost(self, out, Y):
    return np.sum(-Y*np.log(out) - (1-Y)*np.log(1-out))

MultilayerNN._cost = _cost

#%%
def _train_epoch(self,X,Y,LR):
    m = Y.shape[0]
    layers = [None]*self.nlayers
    lin = X
    for i,t in enumerate(self.w):
        lin2 = np.column_stack((np.ones((lin.shape[0],1)),lin))
        out = self._sigmoid(lin2@t)
        layers[i] = (lin,out,lin2)
        lin = out
    
    grad = [None]*self.nlayers
    
    # Output layer
    delta_o = (Y - layers[-1][1])*self._sigmoid(layers[-1][1],deriv=True)
    grad[-1] = (1/m)*(layers[-1][2].transpose()@delta_o)

    for i in range(2,len(layers)+1):
        delta_o = (delta_o@self.w[1-i].transpose())[:,1:]*self._sigmoid(layers[-i][1],deriv=True)
        grad[-i] = (1/m)*(layers[-i][2].transpose()@delta_o)
        
    for i in range(len(layers)):
        self.w[i] += LR*grad[i]
        #print(grad[i])
    
    return out
        
MultilayerNN._train_epoch = _train_epoch

#%%

def predict(self,X):
    lin = X
    for t in self.w:
        lin2 = np.column_stack((np.ones((lin.shape[0],1)),lin))
        lin = self._sigmoid(lin2@t)
    return lin

MultilayerNN.predict = predict

#%%

def save(self,filename):
    with open(filename,'wb') as f:
        pkl.dump(self.w,f)
    
def load(self,filename):
    with open(filename,'rb') as f: 
        self.w = pkl.load(f)
    
MultilayerNN.save = save
MultilayerNN.load = load
    

#%%
    
n = MultilayerNN(784, 10, (256,32))

#%%
j = n.train(trainX, trainY, 0.1, 10000)   
        
