import matplotlib.pyplot as plt
import tensorflow as tf
import networkx as nx
import numpy as np
import math
import time
import os

from model import *
from trainer import *

x_data = np.linspace(-0.5, 0.5, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

x_test = np.linspace(-0.5, 0.5, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_test.shape)
y_test = np.square(x_test) + noise

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
learning_rate = 0.1

with tf.Session() as sess:
    model = Model()
    trainer = Trainer(sess=sess, epoch=4000, print_every=200)    
    predict, loss, record = model.build(x, y)
    feed_dict = {x: x_data, y: y_data}
    feed_dict_test = {x: x_test, y: y_test}
    trainer.train(loss, feed_dict, feed_dict_test, learning_rate, x, y, record)
    trainer.result(x_data, y_data, predict, feed_dict)
    