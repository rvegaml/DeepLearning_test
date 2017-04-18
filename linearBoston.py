'''
File:	linearBoston.py
Author:	Roberto Vega
Description:
	Simple linear model built in tensorflow to classify the boston 
	house-prices dataset.
'''

# Import the required libraries
from __future__ import division
import tensorflow as tf
from sklearn import datasets
from sklearn.preprocessing import scale
import numpy as np

# Load the data
boston = datasets.load_boston()
X = boston.data
Y = boston.target

# Reshape the data
nSamples, nFeat = X.shape
X = np.reshape(X, (nSamples,nFeat))
Y = np.reshape(Y, (nSamples,1))

# Get the zscore of the data
X = scale(X)

# Create the nodes and operations
y = tf.placeholder(tf.float32, shape=(nSamples, 1))

W = tf.Variable(tf.random_uniform([nFeat,1]), name='Weights')
b = tf.Variable(tf.random_uniform([1]), name='Bias')

y_hat = tf.matmul(x, W) + b

# Create the loss function
loss = tf.reduce_sum(tf.square(y - y_hat))

# Create the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.00001)
train = optimizer.minimize(loss)

# Training loop
init = tf.global_variables_initializer()
num_iter = 1000

with tf.Session() as sess:
	sess.run(init)

	for i in range(num_iter):
		sess.run(train, {x:X, y:Y})

	# Evaluate training accuracy
	curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:X, y:Y})
	print("W: %s b: %s training loss: %s"%(curr_W, curr_b, curr_loss))