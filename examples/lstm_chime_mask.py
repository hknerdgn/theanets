#!/usr/bin/env python

import climate
import io
import numpy as np
import theanets
import scipy.io
import os
import tempfile
import urllib
import zipfile
import gzip
import json

init_from_currennt_model = True
current_model_file = 'currennt_trained_network.jsn.gz'

logging = climate.get_logger('lstm-chime')

climate.enable_default_logging()

# choose batch_size to be half the number of sequences in validation set
# so that 2 batches in validation will cover all of it
BATCH_SIZE = 51
TRAIN_NC = os.path.join(tempfile.gettempdir(), 'chime1_train.nc')
VALID_NC = os.path.join(tempfile.gettempdir(), 'chime1_valid.nc')

if not os.path.isfile(TRAIN_NC) or not os.path.isfile(VALID_NC):
    # get the data files from the repository at https://github.com/craffel/lstm_benchmarks
    zipurl = 'https://github.com/craffel/lstm_benchmarks/archive/master.zip'
    logging.info('attempting data copy from url: %s', zipurl)
    z = zipfile.ZipFile(io.BytesIO(urllib.urlopen(zipurl).read()))
    with open(TRAIN_NC, 'wb') as savefile:
        savefile.write(z.read('lstm_benchmarks-master/data/train_1_speaker.nc'))
    with open(VALID_NC, 'wb') as savefile:
        savefile.write(z.read('lstm_benchmarks-master/data/val_1_speaker.nc'))
    z.close()

def load_currennt_model(network=None, filename=None):
# written by M Umut Sen    
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
# written by M Umut Sen    
    num_layers = len(net.layers)
    for lay_ind in np.arange(num_layers-1)+1:
        
        if net.layers[lay_ind].name[0:4] == 'lstm':
            for ind,w in enumerate(net.layers[lay_ind].params):
                net.layers[lay_ind].params[ind].set_value(np.asarray(init_layers[lay_ind-1][w.name[-2:]],dtype=w.get_value().dtype))
        elif net.layers[lay_ind].name[0:3] == 'out':
            net.layers[lay_ind].params[0].set_value(np.asarray(init_layers[lay_ind-1]['W'],dtype=net.layers[lay_ind].params[0].get_value().dtype))
            net.layers[lay_ind].params[1].set_value(np.asarray(init_layers[lay_ind-1]['b'],dtype=net.layers[lay_ind].params[1].get_value().dtype))

def batch_at(features, labels, seqBegins, seqLengths):
    maxSeqLength = np.max(seqLengths)
    batchSize = len(seqBegins)
    ltype, lshape = 'int32', (maxSeqLength, batchSize)
    mtype, mshape = 'int32', (maxSeqLength, batchSize)
    ftype, fshape = 'f', (maxSeqLength, batchSize, features.shape[-1])
    f = np.zeros(fshape, dtype=ftype)
    l = np.zeros(lshape, dtype=ltype)
    m = np.zeros(mshape, dtype=mtype)
    for b in range(batchSize):
	sb=seqBegins[b]
	sl=seqLengths[b]
        f[0:sl, b] = features[sb:sb+sl]
        l[0:sl, b] = labels[sb:sb+sl]
        m[0:sl, b] = np.ones(sl)
    return [f,l,m]

# returns a callable that chooses sequences from netcdf data
# the callable (sample) does random sequence shuffling without replacement
# or can get deterministic ordered batches
# circles back to choose from all the sequences once the unchosen set becomes zero size
def batches(dataset, choose_random=True):
    steps = dataset.dimensions['numTimesteps']
    seqLengths = dataset.variables['seqLengths'].data;
    seqBegins = np.concatenate(([0],np.cumsum(seqLengths)[:-1]))
    numSeqs = len(seqLengths)
    def sample(st={"unchosen":np.arange(numSeqs)}):
        unchosen = st["unchosen"]
	lu = len(unchosen)
        if (lu == 0):
            unchosen = np.arange(numSeqs)
        elif (lu < BATCH_SIZE):
            rest = np.setdiff1d(np.arange(numSeqs),unchosen)
            add = np.random.choice(rest, BATCH_SIZE - lu, replace=False)
            unchosen = np.union1d(unchosen, add)
        if choose_random:
            chosen = np.random.choice(unchosen, BATCH_SIZE, replace=False)
	else:
            chosen = unchosen[np.arange(BATCH_SIZE)]
        st["unchosen"] = np.setdiff1d(unchosen,chosen)
#        print chosen
        [f,l,m] = batch_at(dataset.variables['inputs'].data, dataset.variables['targetClasses'].data, seqBegins[chosen], seqLengths[chosen])
        return [f,l,m]
    return sample

e = theanets.Experiment(
    theanets.maskedrecurrent.Classifier,
    layers=(39, ('lstm',100), ('lstm' , 100),  51),
    recurrent_error_start=0,
    batch_size=BATCH_SIZE,
    optimize='sgd',
    input_noise=0.0,
    learning_rate=0.01,
#    max_gradient_norm=10,
#    truncate_gradient=25,
#    gradient_clip=10,
    patience=50,
    validate_every=1,
    valid_batches=2,
    train_batches=20
)

if (init_from_currennt_model):
    currennt_trained_model = load_currennt_model(filename=current_model_file)
    init_theanets_model(net = e.network, init_layers=currennt_trained_model)

e.train(batches(scipy.io.netcdf_file(open(TRAIN_NC))),
        batches(scipy.io.netcdf_file(open(VALID_NC)), choose_random=False))
