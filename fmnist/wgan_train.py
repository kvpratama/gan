# GAN Architecture based on https://github.com/wiseodd/generative-models/tree/master/GAN/wasserstein_gan

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets import base
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


mb_size = 32
X_dim = 784
z_dim = 10
h_dim = 128
logs_path = 'tensorboard/wgan'
_SAVE_PATH = 'checkpoints/fmnist'


'''If dataset not available in the input path download it.'''

base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
file_names = ['train-images-idx3-ubyte.gz',
          'train-labels-idx1-ubyte.gz',
          't10k-images-idx3-ubyte.gz',
          't10k-labels-idx1-ubyte.gz']

print('Maybe will download the dataset, this can take a while')
for name in file_names:
    base.maybe_download(name, 'dataset/', base_url + name)

fmnist = input_data.read_data_sets(os.getenv("HOME") + '/Documents/data_set/fmnist/', one_hot=True)

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

X = tf.placeholder(tf.float32, shape=[None, X_dim])

D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out


G_sample = generator(z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(-D_loss, var_list=theta_D))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(G_loss, var_list=theta_G))

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

# -------------- TensorBoard summaries -----------------

summ_D_loss = tf.summary.scalar("D_loss", D_loss)

summ_G_loss = tf.summary.scalar("G_loss", G_loss)

# -------------- TensorBoard summaries -----------------

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# create summary writer
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

saver = tf.train.Saver()

if not os.path.exists('wgan-out/'):
    os.makedirs('wgan-out/')
    
if not os.path.exists('checkpoints/'):
    os.makedirs('checkpoints/')

i = 0

for it in range(1000000):
    for d_it in range(5):
        X_mb, _ = fmnist.train.next_batch(mb_size)

        _, D_loss_curr, _, summary = sess.run(
            [D_solver, D_loss, clip_D, summ_D_loss],
            feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
        )
        
        if d_it == 4:
            summary_writer.add_summary(summary, it)

    _, G_loss_curr, summary = sess.run(
        [G_solver, G_loss, summ_G_loss],
        feed_dict={z: sample_z(mb_size, z_dim)}
    )
    summary_writer.add_summary(summary, it)

    if it % 1000 == 0:
        print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss_curr, G_loss_curr))
        
        samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})

        fig = plot(samples)
        plt.savefig('wgan-out/{}.png'
                    .format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
        
        saver.save(sess, save_path=_SAVE_PATH + "/checkpoint", global_step=i)
