import math
import tensorflow as tf
import numpy as np
import pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import json
import argparse
import multiprocessing as mp
import time


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

NUM_FEATURES = 21
NUM_CLASSES = 3
hidden1_units = 10
hidden2_units = 10
    
seed = 10
learning_rate = 0.01
no_epochs = 100

tf.set_random_seed(seed)
np.random.seed(seed)

# initialization routines for bias and weights

def init_bias(n = 1):
    return(tf.Variable(np.zeros(n), dtype=tf.float32))
        
def init_weights(n_in=1, n_out=1, logistic=True):
    W_values = np.asarray(np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)), size=(n_in, n_out)))
    if logistic == True:
        W_values *= 4
        return(tf.Variable(W_values, dtype=tf.float32))

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

def mln(x):
    
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = init_weights(NUM_FEATURES, hidden1_units)
        biases = init_bias(hidden1_units)
        #tf.cast(x, tf.float32)
        hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)

        # Hidden 2
    with tf.name_scope('hidden2'):
        weights = init_weights(hidden1_units, hidden2_units)
        biases = init_bias(hidden2_units)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
        
      # Linear
    with tf.name_scope('softmax_linear'):
        weights = init_weights(hidden2_units, NUM_CLASSES)
        biases = init_bias(NUM_CLASSES)
        logits = tf.matmul(hidden2, weights) + biases
        
    return logits

def train(batch_size):
    train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter= ',')
    trainX, train_Y = train_input[1:, :21], train_input[1:,-1].astype(int) #trainX=all rows(except first=name) and 0th-20th columns(features),train_Y (d=target output)= all rows (except first=name) of last col
    trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0)) #normalize data

    trainY = np.zeros((train_Y.shape[0], NUM_CLASSES)) #create a 2126 x 3 matrix of values=0. 2126=total no. of rows(input data), 3=no. of classes
    trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix. train_Y.shape[0]=2126


    # experiment with small datasets
    trainX = trainX[:1000] #first 1000 col of data
    trainY = trainY[:1000] #first 1000 col of data

    n = trainX.shape[0] #n=1000

    #train test split
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.3)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    #5-fold cross validation
    kf = KFold(n_splits=5) # Define the split - into 5 folds 
    print(kf.get_n_splits(X_train)) # returns the number of splitting iterations in the cross-validator

    print(kf)#KFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X_train):
        x_tr, x_te = X_train[train_index], X_train[test_index]
        y_tr, y_te = y_train[train_index], y_train[test_index]
        #print('X TRAIN VALUES:',x_tr)
        #print('X TEST VALUES:',x_te)
    print(x_tr.shape)
    print(y_tr.shape)
    print(x_te.shape)
    print(y_te.shape)


    # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    print(x.shape)
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES]) #d?

    y = mln(x)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
        cross_entropy = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(cross_entropy, global_step=global_step)

    with tf.name_scope('accuracy'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      correct_prediction = tf.cast(correct_prediction, tf.float32)
      accuracy = tf.reduce_mean(correct_prediction)

    N = len(trainX)
    idx = np.arange(N)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        time_to_update = 0
        for i in range(no_epochs):
            np.random.shuffle(idx)
            trainX = trainX[idx]
            trainY = trainY[idx]

            t = time.time()
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
            time_to_update += time.time() - t
          

            if i%10 == 0:
                test_acc = accuracy.eval(feed_dict={x: x_te, y_: y_te})
                print('batch %d: iter %d, test accuracy %g'%(batch_size, i, test_acc))

        paras = np.zeros(2)
        paras[0] = (time_to_update*1e3)/(no_epochs*(N//batch_size))
        paras[1] = accuracy.eval(feed_dict={x: x_te, y_: y_te})

    return paras


def main():
    batch_sizes = [4,8]
    print('2')
    no_threads = mp.cpu_count()
    p = mp.Pool(processes = no_threads)
    print('3')
    for size in batch_sizes:
        paras = train(size)
    print('7')

    paras = np.array(paras)
    print(paras)
      
    accuracy, time_update = paras[1], paras[0]

    
    accuracy, time_update = [], []
    for batch in batch_sizes:
        test_acc, time_to_update = train(batch)
        accuracy.append(test_acc)
        time_update.append(time_to_update)
    

    # plot learning curves
    print('len of batch sizes',range(len(batch_sizes)))
    plt.figure(1)
    plt.plot(range(len(batch_sizes)), accuracy)
    plt.xticks(range(len(batch_sizes)), batch_sizes)
    plt.xlabel('batch size')
    plt.ylabel('accuracy')
    plt.title('accuracy vs. batch size')
    plt.savefig('./figures/5.5b_1.png')

    plt.figure(2)
    plt.plot(range(len(batch_sizes)), time_update)
    plt.xticks(range(len(batch_sizes)), batch_sizes)
    plt.xlabel('batch size')
    plt.ylabel('time to update (ms)')
    plt.title('time to update vs. batch size')
    plt.savefig('./figures/5.5b_2.png')
 
    plt.show()

main()




  


    
    


        
