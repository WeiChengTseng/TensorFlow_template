import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

class Trainer(object):
    def __init__(self, sess, epoch=None, print_every=100):
        """
        Initialize Trainer variables
        
        Input:
        - sess: tf.sess declared outside Trainer
        - epoch: the number of epoch we need to train
        - print_every: how often we print the result
        """
        
        self.epoch = epoch
        self.print_every = print_every
        self.sess = sess
        return

    def train(self, objective_fun, feed_dict, feed_dict_test, learn_r, x, y, record):
        """
        Train the model.

        Input:
        - objective_fun: the objective function of the model.
        - feed_dict: feed_dict which contains trzining data.
        - fedd_dict_test: feed_dict which contains testing data.
        - learn_r: learning rate of optimization.
        - x: input placeholder.
        - y: ground truth placeholder.
        - record: the record of training on testing data.
        """
        merged = tf.summary.merge_all()
        writer_train = tf.summary.FileWriter('logs/train/', self.sess.graph)
        writer_test = tf.summary.FileWriter('logs/test/')
        self.sess.run(tf.global_variables_initializer())

        train_step = tf.train.GradientDescentOptimizer(learn_r).minimize(objective_fun)
        for epoch in range(self.epoch):
            summary_train, _ = self.sess.run([merged, train_step], feed_dict=feed_dict)
            if epoch % self.print_every == 0:
                loss = self.sess.run(objective_fun, feed_dict=feed_dict)
                print('loss:', loss)
            summary_test = self.sess.run(record, feed_dict=feed_dict_test)

            writer_train.add_summary(summary_train, epoch)
            writer_test.add_summary(summary_test, epoch)
        return

    def result(self, x, y, prediction, feed_dict):
        """
        Show the result.
        - x: input data
        - y: output data
        - prediction: prediction tensor
        - feed_dict: feed_dict to sess.run
        """
        y_pred = self.sess.run(prediction, feed_dict=feed_dict)     
        plt.plot(x, y, 'o', x, y_pred, lw = 3)
        plt.savefig('result/result.png')

        return

