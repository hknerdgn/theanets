#!/usr/bin/env python

import matplotlib.pyplot as plt
import theanets

from utils import load_mnist, plot_layers


train, valid, _ = load_mnist(labels=True)

N = 16

e = theanets.Experiment(
    theanets.Classifier,
    layers=(784, N * N, 10),
    train_batches=100,
)
e.train(train, valid)

plot_layers([e.network.get_weights(1)])
plt.tight_layout()
plt.show()
