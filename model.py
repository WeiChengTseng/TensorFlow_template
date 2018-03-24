import tensorflow as tf
import numpy as np
import os

class Model():
    def __init__(self):
        return

    def build(self, x, y):
        """
        Build the model.

        Input:
        - x: the input data placeholder.
        - y: ground truth placeholder.
        Output:
        - prediction: the result of the logit regression.
        - loss: loss of the model
        - record: the record of training on testing data
        """

        logits = self.sub_model(x)
        prediction = self.predict(logits)
        loss, record = self.loss(y, prediction)

        return prediction, loss, record

    def sub_model(self, x):
        """
        Define the architecture of the model.

        Input:
        - x: the input data, that is, the peptide sequences.
        Output:
        - logits: the result of the logit regression.
        """

        with tf.name_scope('w_b'):
            W_L1 = tf.Variable(tf.random_normal([1, 10]))
            W_L2 = tf.Variable(tf.random_normal([10, 1]))
            b_L1 = tf.Variable(tf.zeros([1, 10]))
            b_L2 = tf.Variable(tf.zeros([1, 1]))
            self.variable_summaries(W_L1)
            self.variable_summaries(W_L2)
            self.variable_summaries(b_L1)
            self.variable_summaries(b_L2)
        with tf.name_scope('layer_1'):
            h1 = tf.matmul(x ,W_L1) + b_L1 
        with tf.name_scope('tanh'):
            L1 = tf.nn.tanh(h1)  
        with tf.name_scope('layer_2'):    
            logits = tf.matmul(L1 ,W_L2) + b_L2

        return logits

    def predict(self, logits):
        """
        Predict the labels according to the model.

        Input:
        - logits: the result of the logit regression.

        Output:
        - prediction: the result of the prediction
        """
        
        prediction = tf.nn.tanh(logits)

        return prediction

    def loss(self, labels, prediction):
        """
        Define the loss of the model.

        Input:
        - label: the ground truth of the prediction.
        - prediction: prediction of the model.

        Output:
        - loss: the loss of the model.
        - logits: the result of the logit regression.
        """

        loss = tf.reduce_mean(tf.square(labels - prediction))
        record = tf.summary.scalar('loss', loss)

        return loss, record
    
    def variable_summaries(self, var):
        """
        Define the tensorboard scalar and histogram summary.

        Input:
        - var: the variable we want to summarize in tensorboard.
        """
        with tf.name_scope("summaries"):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
                tf.summary.scalar('stddev',stddev)
                tf.summary.scalar('max',tf.reduce_max(var))
                tf.summary.scalar('min',tf.reduce_min(var))
                tf.summary.histogram('histogram',var)
    