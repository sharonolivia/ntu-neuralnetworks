#
# Project 2 Part a
#

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import train_test_split

import os
if not os.path.isdir('figures_partA'):
    print('creating the figures folder for part a')
    os.makedirs('figures_partA')


NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 100
batch_size = 128


seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')
    
    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    
    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels-1] = 1

    return data, labels_


# x= [no_patterns, input_width, input_height, no_input_maps]
# w = [window_width, window_height, input_maps, output_maps]

def cnn(images,numMaps1, numMaps2):

    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
    
    #Conv 1 and poo1 1
    W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, numMaps1], stddev=1.0/np.sqrt(NUM_CHANNELS*9*9)), name='weights_1')
    b1 = tf.Variable(tf.zeros([numMaps1]), name='biases_1')

    conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1) # stride = [1, 1, 1, 1]
    pool_1 = tf.nn.max_pool(conv_1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')
    # pool_1.get_shape() = [?,12,12,50]

    #Conv 2 and pool 2
    W2 = tf.Variable(tf.truncated_normal([5, 5, numMaps1, numMaps2], stddev=1.0/np.sqrt(NUM_CHANNELS*5*5)), name='weights_2')
    b2 = tf.Variable(tf.zeros([numMaps2]), name='biases_2')

    conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2) # stride = [1, 1, 1, 1]
    pool_2 = tf.nn.max_pool(conv_2, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_2')

    # FC size: [1 x 1 x num_classes]

    #fully connected layer
    dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value # dim = 7200
    reshape = tf.reshape(pool_2, [-1, dim]) # reshape.get_shape() = (?, 7200) # flatten to 1D array???
    
    w = tf.Variable(tf.truncated_normal([dim, 300], stddev=1.0 / np.sqrt(dim), name="weights3"))
    b = tf.Variable(tf.zeros([300]), name = "biases_3")
    fc1 = tf.nn.relu(tf.matmul(reshape, w) + b, name= "fc1") # fc1.get_shape() = (?,300)

    keep_prob=tf.placeholder(tf.float32)
    fc1_drop=tf.nn.dropout(fc1,keep_prob)


    #Softmax
    W2 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0/np.sqrt(dim)), name='weights_4')
    b2 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')
    logits = tf.add(tf.matmul(fc1, W2), b2, name= "softmax_linear")
    logits_drop=tf.add(tf.matmul(fc1_drop, W2), b2, name= "softmax_linear")


    return logits, logits_drop, conv_1, pool_1, conv_2, pool_2, keep_prob
    

def main(numMaps1, numMaps2):

    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)
    
    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)

    trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    
    logits, logits_drop, conv_1, pool_1, conv_2, pool_2, keep_prob = cnn(x,numMaps1, numMaps2)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    train_step_MO = tf.train.MomentumOptimizer(learning_rate, 0.1).minimize(loss)
    train_step_RMS = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    train_step_AO = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    print('NO. OF FEATURE MAPS 1:',numMaps1)
    print('NO. OF FEATURE MAPS 2:',numMaps2)

    N = len(trainX)
    idx = np.arange(N)

    if numMaps1==70:
        with tf.Session() as sess:
            
            print('momentum...')
            sess.run(tf.global_variables_initializer())

            test_acc_momentum = []
            training_cost_momentum = []
            for i in range(epochs):
              np.random.shuffle(idx)
              trainX, trainY = trainX[idx], trainY[idx]

              for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                  train_step_MO.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
              
              test_acc_momentum.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
              print('iter %d: test accuracy %g'%(i, test_acc_momentum[i]))
              training_cost_momentum.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
              print('iter %d: training cost %g'%(i, training_cost_momentum[i]))

            plt.figure(1)
            plt.title('Test Accuracy vs Epochs (Momentum)')
            mom_label='momentum'
            plt.plot(np.arange(epochs), test_acc_momentum, label=mom_label)
            plt.xlabel('epochs')
            plt.ylabel('test accuracy')
            plt.savefig('./figures_partA/momentum_testacc')

            plt.figure(2)
            plt.title('Training Cost vs Epochs (Momentum)')
            plt.plot(np.arange(epochs), training_cost_momentum, label=mom_label)
            plt.xlabel('epochs')
            plt.ylabel('test accuracy')
            plt.savefig('./figures_partA/momentum_trainingcost')
            

            print('RMSProp ...')
            sess.run(tf.global_variables_initializer())

            test_acc_RMS = []
            training_cost_RMS = []
            for i in range(epochs):
              np.random.shuffle(idx)
              trainX, trainY = trainX[idx], trainY[idx]

              for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                  train_step_RMS.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
              
              test_acc_RMS.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
              print('iter %d: test accuracy %g'%(i, test_acc_RMS[i]))
              training_cost_RMS.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
              print('iter %d: training cost %g'%(i, training_cost_RMS[i]))

            plt.figure(1)
            plt.title('Test Accuracy vs Epochs (RMS Prop)')
            rms_label='RMS Prop'
            plt.plot(np.arange(epochs), test_acc_RMS, label=rms_label)
            plt.xlabel('epochs')
            plt.ylabel('test accuracy')
            plt.savefig('./figures_partA/RMSprop_testacc')

            plt.figure(2)
            plt.title('Training Cost vs Epochs (RMS Prop)')
            plt.plot(np.arange(epochs), training_cost_RMS, label=rms_label)
            plt.xlabel('epochs')
            plt.ylabel('test accuracy')
            plt.savefig('./figures_partA/RMSprop_trainingcost')

            print('Adam ...')
            sess.run(tf.global_variables_initializer())

            test_acc_adam = []
            training_cost_adam = []
            for i in range(epochs):
              np.random.shuffle(idx)
              trainX, trainY = trainX[idx], trainY[idx]

              for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                  train_step_AO.run(feed_dict={x: trainX[start:end], y_: trainY[start:end],})
              
              test_acc_adam.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
              print('iter %d: test accuracy %g'%(i, test_acc_adam[i]))
              training_cost_adam.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
              print('iter %d: training cost %g'%(i, training_cost_adam[i]))

            plt.figure(1)
            plt.title('Test Accuracy vs Epochs (Adam Optimizer)')
            adam_label='Adam Optimizer'
            plt.plot(np.arange(epochs), test_acc_adam, label=adam_label)
            plt.xlabel('epochs')
            plt.ylabel('test accuracy')
            plt.savefig('./figures_partA/adam_testacc')

            plt.figure(2)
            plt.title('Training Cost vs Epochs (Adam Optimizer)')
            plt.plot(np.arange(epochs), training_cost_adam, label=adam_label)
            plt.xlabel('epochs')
            plt.ylabel('test accuracy')
            plt.savefig('./figures_partA/adam_trainingcost')


            print('dropout...')
            sess.run(tf.global_variables_initializer())

            test_acc_dropout = []
            training_cost_dropout = []
            for i in range(epochs):
              np.random.shuffle(idx)
              trainX, trainY = trainX[idx], trainY[idx]

              for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                  train_step.run(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob:0.5})
              
              test_acc_dropout.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
              print('iter %d: test accuracy %g'%(i, test_acc_dropout[i]))
              training_cost_dropout.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
              print('iter %d: training cost %g'%(i, training_cost_dropout[i]))

            plt.figure(1)
            dropout_label='Dropout'
            plt.title('Test Accuracy vs Epochs (Dropout)')
            plt.plot(np.arange(epochs), test_acc_dropout, label=dropout_label)
            plt.xlabel('epochs')
            plt.ylabel('test accuracy')
            plt.savefig('./figures_partA/dropout_testacc')

            plt.figure(2)
            plt.title('Training Cost vs Epochs (Dropout)')
            plt.plot(np.arange(epochs), training_cost_dropout, label=dropout_label)
            plt.xlabel('epochs')
            plt.ylabel('test accuracy')
            plt.savefig('./figures_partA/dropout_trainingcost')
    

def run():
    main(70,90)

        

if __name__ == '__main__':
    #main()
    run()
