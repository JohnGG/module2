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

# Define placholders
inputs = tf.placeholder(shape=(None, 28, 28), name="inputs", dtype=tf.float32)
labels = tf.placeholder(shape=(None, ), name="labels", dtype=tf.int64)

# Define conv layer
filter = tf.get_variable(name="filters", shape=[2,2,1,64])
conv1 = tf.layers.conv2d(
      inputs=tf.reshape(inputs, shape=(-1, 28, 28, 1)),
      filters=32,
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# Define output layer
W = tf.get_variable(name="weights", shape=(6272, 10))
B = tf.get_variable(name="bias2", shape=(10))
logits = tf.matmul(tf.reshape(pool1, shape=(-1, 6272)), W) + B
predictions = tf.argmax(tf.nn.softmax(logits), axis=1)

#Create loss node
one_hot_labels = tf.one_hot(labels, 10)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits))

#Define optimizer ops
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

#Define accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Create arrays for loss and accuracy curves plots
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    for e in range(epochs):
        train_acc = 0
        train_loss = 0
        nb_train_steps = int(len(X_TRAIN) / batch_size)
        for train_step in range(nb_train_steps):
            _, lo, ac = sess.run([optimizer, loss, accuracy],
                                 feed_dict={inputs: X_TRAIN[train_step * batch_size:(train_step + 1) * batch_size],
                                            labels: Y_TRAIN[train_step * batch_size:(train_step + 1) * batch_size]})
            train_acc+=ac
            train_loss+=lo

        train_losses.append(train_loss/nb_train_steps)
        train_accuracies.append(train_acc/nb_train_steps)

        val_acc = 0
        val_loss = 0
        nb_val_steps = int(len(X_VAL) / batch_size)
        for val_step in range(int(len(X_VAL) / batch_size)):
            l, acc = sess.run([loss, accuracy],
                                 feed_dict={inputs: X_VAL[val_step * batch_size:(val_step + 1) * batch_size],
                                            labels: Y_VAL[val_step * batch_size:(val_step + 1) * batch_size]})
            val_acc += acc
            val_loss += l

        val_losses.append(val_loss/nb_val_steps)
        val_accuracies.append(val_acc/nb_val_steps)

        print(val_acc/nb_val_steps)

    # Plot losses
    plt.plot(range(len(val_losses)), val_losses, label="val")
    plt.plot(range(len(train_losses)), train_losses, label="train")
    plt.legend()
    plt.show()

    # Plot accuracies
    plt.plot(range(epochs), val_accuracies, label="val")
    plt.plot(range(epochs), train_accuracies, label="train")
    plt.legend()
    plt.show()