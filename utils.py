import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import numpy as np

class Drawer():
    def __init__(self):
        self.fig = plt.figure(figsize=(4, 4))

    def plot(self, samples):
        self.fig.clf()
        gs = gridspec.GridSpec(4,4)
        gs.update(wspace=0.05, hspace=0.05)

        for i,sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        """
        f,a = plt.subplots(1, 16, figsize=(16,1))
        for i in range(16):
            a[i].imshow(np.reshape(samples[i], (28,28)))
        #f.show()
        """
        return self.fig


def sample_z(mu, logvar):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + (logvar/2) * eps


def kl_divergence_normal_distribution(mu, logvar):
    return tf.reduce_mean(0.5 * tf.reduce_sum(tf.exp(logvar) + mu**2 -1 -logvar, 1))

def get_random_normal(batch_size, dim, mean=0., stddev=1.):
    return np.random.normal(mean, stddev, size=[batch_size, dim])
