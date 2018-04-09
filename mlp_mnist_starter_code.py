import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
epochs = 50
batch_size = 500

# Read dataset
X = np.load("../datasets/MNIST/10k_sample_normal/X.npy")
Y = np.load("../datasets/MNIST/10k_sample_normal/Y.npy")

# Split into train and val datasets
X_TRAIN = X[0:8000]
Y_TRAIN = Y[0:8000]

X_VAL = X[8000:10000]
Y_VAL = Y[8000:10000]

###### MODEL DEFINITION #####

# Define placholders
inputs = tf.placeholder(shape=(None, 28, 28), name="inputs", dtype=tf.float32)
labels = tf.placeholder(shape=(None, ), name="labels", dtype=tf.int64)

#TODO :  Define hidden layer 1

#TODO :  Define hidden layer 2

#TODO :  Define ouput layer

#TODO :  Create loss node

#TODO :  Define optimizer ops

#TODO : define accuracy

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(epochs):
        ###### TRAINING LOOP #####
        # TODO: Mini batch gradient descent (batch size = 500)
        # TODO: Compute accuracy and loss for train dataset after all the optimisation execution are done
        ###### /TRAINING LOOP #####


        ###### EVALUATION LOOP #####
        # TODO: Compute accuracy and loss for val dataset
        ###### /EVALUATION LOOP #####
        pass