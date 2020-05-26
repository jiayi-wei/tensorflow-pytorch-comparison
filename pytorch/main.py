#!/usr/bin/env python

# -*- coding: utf-8 -*-

import torch
import torchvision
import numpy as np
import mnist

n_epochs = 5
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 100

data = mnist.MNIST("../data/")
data.gz = True

train_image, train_label = data.load_training()
test_image, test_label = data.load_testing()
train_size = len(train_image)
test_size = len(test_image)
ds_train = tf.data.Dataset.from_tensor_slices((train_image, train_label))
ds_test = tf.data.Dataset.from_tensor_slices((test_image, test_label))


ds_train = ds_train.map(lambda it1, it2: tf.numpy_function(normalize_img,
                                                           [it1, it2],
                                                           [tf.float32, tf.int32]),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.shuffle(100).batch(batch_size_train)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(lambda it1, it2: tf.numpy_function(normalize_img,
                                                         [it1, it2],
                                                         [tf.float32, tf.int32]),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(batch_size_test)
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


class Net(tf.keras.Model):
    def __init__(self, training=True):
        super(Net, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(10, 5)
        self.conv2 = tf.keras.layers.Conv2D(20, 5)
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.fc1 = tf.keras.layers.Dense(50)
        self.fc2 = tf.keras.layers.Dense(10)

    def call(self, x):
        x = tf.nn.relu(tf.nn.max_pool2d(self.conv1(x),
                                        ksize=2,
                                        strides=2,
                                        padding='VALID'))
        x = self.dropout(self.conv2(x))
        x = tf.nn.relu(tf.nn.max_pool2d(x, ksize=2,
                                        strides=2,
                                        padding='VALID'))
        x = tf.reshape(x, (x.shape[0], -1))
        x = tf.nn.relu(self.fc1(x))
        x = tf.nn.dropout(x, rate=0.2)
        x = tf.nn.softmax(self.fc2(x))
        return x


network = Net()
optim = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                momentum=momentum)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

train_losses = []
test_losses = []


def train(epoch):
    for batch_idx, (data, target) in enumerate(ds_train):
        with tf.GradientTape() as tape:
            output = network(data)
            loss = loss_object(target, output)
        trainable_variables = network.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optim.apply_gradients(zip(gradients, trainable_variables))
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * batch_size_train, train_size,
                    100. * batch_idx * batch_size_train / train_size,
                    loss.numpy()))
            train_losses.append(loss.numpy())
    network.save_weights("./results/model_e{}".format(epoch))


def test():
    test_loss = 0
    correct = 0
    for data, target in ds_test:
        output = network(data)
        test_loss += loss_object(target, output).numpy()
        pred = tf.cast(tf.math.argmax(output, axis=1), dtype=tf.int32)
        correct += tf.math.count_nonzero(tf.math.equal(pred, target)).numpy()
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_size,
        100. * correct / test_size))


test()
for epoch in range(1, n_epochs+1):
    train(epoch)
    test()

np.save('./results/train_loss.npy', train_losses)
np.save('./results/test_loss.npy', test_losses)
