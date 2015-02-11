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

from lstm_init_currennt import load_currennt_model
from lstm_init_currennt import init_theanets_model

init_from_currennt_model = True

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
    layers=(39, ('lstm',78), ('lstm' , 100),  51),
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
    currennt_trained_model = load_currennt_model(filename='model_currennt/trained_network2.jsn.gz')
    init_theanets_model(net = e.network, init_layers=currennt_trained_model)

e.train(batches(scipy.io.netcdf_file(open(TRAIN_NC))),
        batches(scipy.io.netcdf_file(open(VALID_NC)), choose_random=False))
