#!/usr/bin/env python

import json
import gzip
import numpy as np

import theanets
import os


def sigm(x):
    return 1/(1+np.exp(-x))

def softmax_row(w):
    e = np.exp(np.array(w))
    return e / np.sum(e,axis=0)
    
def softmax(w):
    e = np.exp(np.array(w))
    return e / np.sum(e)
    
class LstmCPU_Simple:
    
    def __init__(self,size_in,size_hidden,size_out):
        
        self.size_in = size_in
        self.size_hidden = size_hidden
        self.size_out = size_out
        
    def initialize(self, currentModel=None, std=0.1):
        
        if currentModel is None:
        
            self.Wxc = np.asarray(std*np.random.randn(self.size_in,self.size_hidden),np.float32)
            self.Wxi = np.asarray(std*np.random.randn(self.size_in,self.size_hidden),np.float32)
            self.Wxf = np.asarray(std*np.random.randn(self.size_in,self.size_hidden),np.float32)
            self.Wxo = np.asarray(std*np.random.randn(self.size_in,self.size_hidden),np.float32)

            self.bc = np.asarray(std*np.zeros(self.size_hidden),np.float32)
            self.bi = np.asarray(std*np.zeros(self.size_hidden),np.float32)
            self.bf = np.asarray(std*np.zeros(self.size_hidden),np.float32)
            self.bo = np.asarray(std*np.zeros(self.size_hidden),np.float32)
            
            self.Whc = np.asarray(std*np.random.randn(self.size_hidden,self.size_hidden),np.float32)
            self.Whi = np.asarray(std*np.random.randn(self.size_hidden,self.size_hidden),np.float32)
            self.Whf = np.asarray(std*np.random.randn(self.size_hidden,self.size_hidden),np.float32)
            self.Who = np.asarray(std*np.random.randn(self.size_hidden,self.size_hidden),np.float32)
            
            self.ci = np.asarray(std*np.random.randn(self.size_hidden),np.float32)
            self.cf = np.asarray(std*np.random.randn(self.size_hidden),np.float32)
            self.co = np.asarray(std*np.random.randn(self.size_hidden),np.float32)
            
            self.Wout = np.asarray(std*np.random.randn(self.size_hidden,self.size_out),np.float32)
            self.bout = np.asarray(std*np.zeros(self.size_out),np.float32)
        else:
            
            self.Wxc = currentModel['xc']
            self.Wxi = currentModel['xi']
            self.Wxf = currentModel['xf']
            self.Wxo = currentModel['xo']
            
            self.bc = currentModel['bc']
            self.bi = currentModel['bi']
            self.bf = currentModel['bf']
            self.bo = currentModel['bo']
            
            # self.Whc = np.transpose(currentModel['hc'])
            # self.Whi = np.transpose(currentModel['hi'])
            # self.Whf = np.transpose(currentModel['hf'])
            # self.Who = np.transpose(currentModel['ho'])
            
            self.Whc = currentModel['hc']
            self.Whi = currentModel['hi']
            self.Whf = currentModel['hf']
            self.Who = currentModel['ho']
            
            self.ci = currentModel['ci']
            self.cf = currentModel['cf']
            self.co = currentModel['co']
            
            self.Wout = currentModel['Wout']
            self.bout = currentModel['bout']
            
        # self.Wx = np.
    
    def forward(self,feats):
        probs = []
        Hs = []
        for i in np.arange(len(feats)):
            X_now = feats[i]
            len_now = X_now.shape[1]
            
            I_now = np.zeros((self.size_hidden,len_now),np.float32)
            F_now = np.zeros((self.size_hidden,len_now),np.float32)
            O_now = np.zeros((self.size_hidden,len_now),np.float32)
            H_now = np.zeros((self.size_hidden,len_now),np.float32)
            C_now = np.zeros((self.size_hidden,len_now),np.float32)
            Y_now = np.zeros((self.size_out,len_now),np.float32)
            # Ct_now = np.zeros((self.size_hidden,len_now),np.float32)
            
            Wix_now = np.dot(np.transpose(self.Wxi),X_now)
            Wcx_now = np.dot(np.transpose(self.Wxc),X_now)
            Wfx_now = np.dot(np.transpose(self.Wxf),X_now)
            Wox_now = np.dot(np.transpose(self.Wxo),X_now)
            
            #t=0:
            #print Wix_now[:,0].shape, len(self.bi), I_now.shape
            I_now[:,0] = sigm(Wix_now[:,0] + self.bi )
            F_now[:,0] = sigm(Wfx_now[:,0] + self.bf )
            
            # Ct_now[:,0] = np.tanh(Wcx_now[:,0] + self.bc)
            C_now[:,0] = I_now[:,0] * np.tanh(Wcx_now[:,0] + self.bc)
            
            O_now[:,0] = sigm(Wox_now[:,0] + self.bo + self.co*C_now[:,0] )
            
            H_now[:,0] = O_now[:,0] * np.tanh(C_now[:,0])
            
            Y_now[:,0] = softmax(np.dot(np.transpose(self.Wout),H_now[:,0]) + self.bout)
            
            for t in np.arange(len_now-1)+1:
                I_now[:,t] = sigm(Wix_now[:,t] + np.dot(np.transpose(self.Whi),H_now[:,t-1]) + self.bi + self.ci*C_now[:,t-1] )
                F_now[:,t] = sigm(Wfx_now[:,t] + np.dot(np.transpose(self.Whf),H_now[:,t-1]) + self.bf + self.cf*C_now[:,t-1] )
                
                # Ct_now[:,t] = np.tanh(Wcx_now[:,t] + np.dot(np.transpose(self.Whc),H_now[:,t-1]) + self.bc )
                C_now[:,t] = F_now[:,t]*C_now[:,t-1] + I_now[:,t]*np.tanh(Wcx_now[:,t] + np.dot(np.transpose(self.Whc),H_now[:,t-1]) + self.bc )
                
                O_now[:,t] = sigm(Wox_now[:,t] + np.dot(np.transpose(self.Who),H_now[:,t-1]) + self.bo + self.co*C_now[:,t] )
                
                H_now[:,t] = O_now[:,t] * np.tanh(C_now[:,t])
            
                Y_now[:,t] = softmax(np.dot(np.transpose(self.Wout),H_now[:,t]) + self.bout)
            
            probs.append(Y_now)
            Hs.append(H_now)
        
        return probs, Hs
        
    def predict(self,probs):
        Yhat = []
        for i in np.arange(len(probs)):
            probs_now = probs[i]
            Yhat.append(probs_now.argmax(axis=0))
        return Yhat
            
    def acc(self,Y,Yhat):
        numTrue = 0
        numTotal = 0
        for i in np.arange(len(Y)):
            numTrue = numTrue + (Y[i]==Yhat[i]).sum()
            numTotal = numTotal + Y[i].shape[0]
        return float(numTrue)/numTotal

def load_currennt_model(network=None, filename=None):
    
    if network is None:
        if filename[-2:] == 'gz':
            with gzip.open(filename) as f:
                    cont = f.read()
        else:
            with open(filename) as f:
                    cont = f.read()
        network = json.loads(cont)

    layers = network['layers']
    weights = network['weights']

    out = []
    
    for l in layers:
        L = l['size']
        name = l['name']
        layernow = {}
        layernow['type'] = l['type']
        
        if l['type'] == 'lstm':
            ws_input = weights[name]['input']
            ws_bias = weights[name]['bias']
            ws_internal = weights[name]['internal']
            
            PL = P*L
                        
            layernow['xc'] = np.transpose(np.reshape(ws_input[0:PL],(L,P)))
            layernow['xi'] = np.transpose(np.reshape(ws_input[PL:2*PL],(L,P)))
            layernow['xf'] = np.transpose(np.reshape(ws_input[2*PL:3*PL],(L,P)))
            layernow['xo'] = np.transpose(np.reshape(ws_input[3*PL:4*PL],(L,P)))

            layernow['xh'] = np.hstack((layernow['xi'],layernow['xf'],layernow['xc'],layernow['xo']))
            
            layernow['bc'] = ws_bias[0:L]
            layernow['bi'] = ws_bias[L:2*L]
            layernow['bf'] = ws_bias[2*L:3*L]
            layernow['bo'] = ws_bias[3*L:4*L]

            layernow['_b'] = np.hstack((layernow['bi'],layernow['bf'],layernow['bc'],layernow['bo']))
            
            layernow['hc'] = np.transpose(np.reshape(ws_internal[0:L*L],(L,L)))
            layernow['hi'] = np.transpose(np.reshape(ws_internal[L*L:2*L*L],(L,L)))
            layernow['hf'] = np.transpose(np.reshape(ws_internal[2*L*L:3*L*L],(L,L)))
            layernow['ho'] = np.transpose(np.reshape(ws_internal[3*L*L:4*L*L],(L,L)))

            layernow['hh'] = np.hstack((layernow['hi'],layernow['hf'],layernow['hc'],layernow['ho']))
            
            layernow['ci'] = ws_internal[4*L*L:4*L*L+L]
            layernow['cf'] = ws_internal[4*L*L+L:4*L*L+2*L]
            layernow['co'] = ws_internal[4*L*L+2*L:4*L*L+3*L]
                        
        elif l['type'] == 'softmax':
            layernow['W'] = np.transpose(np.reshape(weights[name]['input'],(L,P)))
            layernow['b'] = weights[name]['bias']

        P = L
        
        if l['type'] == 'lstm' or l['type'] == 'softmax':
            out.append(layernow)
    
    return out

def init_theanets_model(net, init_layers):
    num_layers = len(net.layers)
    for lay_ind in np.arange(num_layers-1)+1:
        
        if net.layers[lay_ind].name[0:4] == 'lstm':
            for ind,w in enumerate(net.layers[lay_ind].params):
                net.layers[lay_ind].params[ind].set_value(np.asarray(init_layers[lay_ind-1][w.name[-2:]],dtype=w.get_value().dtype))
                
        #    for ind,b in enumerate(net.layers[lay_ind].biases):
        #        net.layers[lay_ind].biases[ind].set_value(np.asarray(init_layers[lay_ind-1][b.name[-2:]],dtype=b.get_value().dtype))
                
        elif net.layers[lay_ind].name[0:3] == 'out':
            net.layers[lay_ind].params[0].set_value(np.asarray(init_layers[lay_ind-1]['W'],dtype=net.layers[lay_ind].params[0].get_value().dtype))
            net.layers[lay_ind].params[1].set_value(np.asarray(init_layers[lay_ind-1]['b'],dtype=net.layers[lay_ind].params[1].get_value().dtype))

def list2bulk(X,Y):
    numSeq = len(X)
    
    maxLen = 0
    for i in np.arange(len(X)):
        maxLen = np.max((maxLen,X[i].shape[1]))
    
    print "max length", maxLen
    print "num sequences", numSeq
    
    dim = X[0].shape[0]
    
    X_bulk = np.zeros((dim,numSeq*maxLen))
    Y_bulk = np.zeros(numSeq*maxLen)
    
    for i in np.arange(len(X)):
        for j in np.arange(maxLen):
            if j < X[i].shape[1]:
                # print i,j,X[i].shape, X_bulk.shape
                X_bulk[:,j*numSeq+i] = X[i][:,j]
                Y_bulk[j*numSeq+i] = Y[i][j]
    
    return X_bulk, Y_bulk, numSeq


## starts here:

def list2bulk_2(X,Y,maxLen=50,numSeq=None):
    
    if numSeq is None:
        numSeq = len(X)
    
    # maxLen = 0
    # for i in np.arange(len(X)):
        # maxLen = np.max((maxLen,X[i].shape[1]))
    
    print "max length", maxLen
    print "num sequences", numSeq
    
    dim = X[0].shape[0]
    
    X_bulk = np.asarray(np.zeros((maxLen,numSeq,dim)),np.float32)
    Y_bulk = np.asarray(np.zeros((maxLen,numSeq)),np.int32)
    
    for seq_in in np.arange(numSeq):
        for time_in in np.arange(maxLen):
            X_bulk[time_in,seq_in,:] = X[seq_in][:,time_in]
            Y_bulk[time_in,seq_in] = Y[seq_in][time_in]
            
    return X_bulk, Y_bulk, numSeq
