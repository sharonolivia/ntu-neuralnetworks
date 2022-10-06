#
# Project 1, starter code part a
#
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

epochs = 50

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)


def mln(batch_size,num_hidden):
    #print('4')
    NUM_FEATURES = 21
    NUM_CLASSES = 3
    NUM_HIDDEN = 10

    lr = 0.01
    
    #batch_size = 32
    num_neurons = 10
    seed = 10
    np.random.seed(seed)

    #read train data

    train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter= ',')
    trainX, train_Y = train_input[1:, :21], train_input[1:,-1].astype(int) #trainX=all rows(except first=name) and 0th-20th columns(features),train_Y (d=target output)= all rows (except first=name) of last col
    trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0)) #normalize data

    trainY = np.zeros((train_Y.shape[0], NUM_CLASSES)) #create a 2126 x 3 matrix of values=0. 2126=total no. of rows(input data), 3=no. of classes
    trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 #one hot matrix. train_Y.shape[0]=2126

    # initialization routines for bias and weights
    def init_bias(n = 1):
        return(tf.Variable(np.zeros(n), dtype=tf.float32))
        
    def init_weights(n_in=1, n_out=1, logistic=True):
        W_values = np.asarray(np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                                high=np.sqrt(6. / (n_in + n_out)),
                                                size=(n_in, n_out)))
        if logistic == True:
            W_values *= 4
        return(tf.Variable(W_values, dtype=tf.float32))


    # experiment with small datasets
    trainX = trainX[:1000] #first 1000 col of data
    trainY = trainY[:1000] #first 1000 col of data

    n = trainX.shape[0] #n=1000

    #train test split
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.3)


    #5-fold cross validation
    kf = KFold(n_splits=5) # Define the split - into 5 folds 
    print(kf.get_n_splits(X_train)) # returns the number of splitting iterations in the cross-validator

    print(kf)#KFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X_train):
        x_tr, x_te = X_train[train_index], X_train[test_index]
        y_tr, y_te = y_train[train_index], y_train[test_index]
        #print('X TRAIN VALUES:',x_tr)
        #print('X TEST VALUES:',x_te)


    # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    print(x.shape)
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES]) #d?


    #Define variables:
    V = init_weights(NUM_HIDDEN, NUM_CLASSES) #weight of output layer
    c = init_bias(NUM_CLASSES) #bias of output layer
    W = init_weights(NUM_FEATURES, NUM_HIDDEN) #weight of hidden layer
    b = init_bias(NUM_HIDDEN) #bias of hidden layer


    #CHANGE add relu and softmax function
    z = tf.matmul(x, W) + b #synaptic input to hidden layer
    h = tf.nn.relu(z) #output of hidden layer
    y = tf.matmul(h, V) + c #synaptic input to output(last/softmax) layer

    cost = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y),axis=1))

    #optimizer
    grad_u = -(y_ - y)
    grad_V = tf.matmul(tf.transpose(h), grad_u)
    grad_c = tf.reduce_sum(grad_u, axis=0)

    y_h = h*(1-h)
    grad_z = tf.matmul(grad_u, tf.transpose(V))*y_h
    grad_W = tf.matmul(tf.transpose(x), grad_z)
    grad_b = tf.reduce_sum(grad_z, axis=0)

    W_new = W.assign(W - lr*grad_W)
    b_new = b.assign(b - lr*grad_b)
    V_new = V.assign(V - lr*grad_V)
    c_new = c.assign(c - lr*grad_c)

    '''
    # Build the graph for the deep net
            
    weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, NUM_CLASSES], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weights')
    biases  = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
    logits  = tf.matmul(x, weights) + biases
    '''

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_)
    loss = tf.reduce_mean(cross_entropy)

    class_error = tf.reduce_sum(tf.cast(tf.not_equal(y > 0.5, y_tr), tf.int32))

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)


    N = len(x_tr)
    idx = np.arange(N)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_acc = []
        test_acc = []
        class_err = []
        #cross_entropy_err = []
        time_to_update = 0
        paras = np.zeros(4)
        
        W_, b_ = sess.run([W, b])
        #print('W: {}, b: {}'.format(W_, b_))
        V_, c_ = sess.run([V, c])
        #print('V:{}, c:{}'.format(V_, c_))
        
        
        for i in range(epochs):
            train_op.run(feed_dict={x: x_tr, y_: y_tr})
            

            #err_=0

            if i == 0:
                z_, h_, y2, class_err_, grad_u_, dh_, grad_z_, grad_V_, grad_c_, grad_W_, grad_b_ = sess.run(
                    [ z, h, y, class_error, grad_u, y_h, grad_z, grad_V, grad_c, grad_W, grad_b], {x:x_tr, y_:y_tr})
                '''
                print('iter: {}'.format(i+1))
                print('z: {}'.format(z_))
                print('h: {}'.format(h_))
                print('y: {}'.format(y2))
                print('grad_u: {}'.format(grad_u_))
                print('dh: {}'.format(dh_))
                print('grad_z:{}'.format(grad_z_))
                print('grad_V:{}'.format(grad_V_))
                print('grad_c:{}'.format(grad_c_))
                print('grad_W:{}'.format(grad_W_))
                print('grad_b:{}'.format(grad_b_))
                print('class error: {}'.format(class_err_))
                '''
                

            train_acc.append(accuracy.eval(feed_dict={x:x_tr, y_:y_tr}))
            test_acc.append(accuracy.eval(feed_dict={x:x_te, y_:y_te}))

           
            sess.run([W_new, b_new, V_new, c_new], {x:x_tr, y_:y_tr})
            class_err_ = sess.run(class_error, {x:x_tr, y_:y_tr})
            class_err.append(class_err_)
            
            # shuffle data
            np.random.shuffle(idx)
            trainX = x_tr[idx]
            trainY = y_tr[idx]
            
            t = time.time()
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
            time_to_update += time.time() - t
            
            
            if i % 100 == 0:
                print('iter %d: train accuracy %g'%(i, train_acc[i]))
                #test_acc = accuracy.eval(feed_dict={x: x_te, y_: y_te})
                print('batch %d: iter %d, test accuracy %g'%(batch_size, i, test_acc[i]))

        
            
            paras[1] = accuracy.eval(feed_dict={x:x_te, y_:y_te}) #test accuracy
            paras[2] = accuracy.eval(feed_dict={x:x_tr, y_:y_tr}) #train accuracy
            paras[3] = class_err_ #class error

        timer = (time_to_update*1e3)/(epochs*(N//batch_size))
        print('TRAIN ACC:', train_acc)        
    return train_acc, test_acc, timer, class_err
    

def main():
    
    #def plot_error_graph(err_batch):
        #error_graph(err_batch, epochs)
    batch_sizes = [4,8,16,32,64]
    num_neurons=[5,10,15,20,25]
    no_threads = mp.cpu_count()
    p = mp.Pool(processes = no_threads)
    
    
    time_batch=[]
    time=[]
    
    #time_update = []
    for num in num_neurons:
        tr_accuracy, te_accuracy=[], []
        err_batch=[]
        train_acc, test_acc, time, class_err = mln(64,num)
        for i in range(epochs):
            #train_acc, test_acc, time, class_err = mln(size) = mln(size)
            tr_accuracy.append(train_acc)
            te_accuracy.append(test_acc)
            '''
            if size == 32:
                err_batch.append(class_err)
            '''
        '''
        if size == 32:
            error_graph(class_err, epochs)
        '''
                
        print('train acc:', tr_accuracy)
        accuracy_graphs(test_acc, train_acc, epochs)
       
    '''
    time_update=[]
    for size in batch_sizes:
        train_acc, test_acc, time, class_err = mln(size)
        time_update.append(time)
    time_graph(time_update,batch_sizes)
    '''
    

    
# train, test accuracies vs epoch graphs
def accuracy_graphs(test_accuracy, train_accuracy, epochs):

        
    # train acc vs epoch
    print('no. of epochs',epochs)
    plt.figure(1)
    plt.plot(range(epochs), train_accuracy)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy vs. epoch')

    # test acc vs epoch
    print('no. of epochs',epochs)
    plt.figure(2)
    plt.plot(range(epochs), test_accuracy)
    plt.xlabel('epoch')
    plt.ylabel('test accuracy')
    plt.title('test accuracy vs. epoch')


    plt.show()

# class error vs epoch
def error_graph(c_err, epochs):
    plt.figure(1)
    print('no. of epochs',epochs)
    plt.figure(3)
    plt.plot(range(epochs), c_err)
    plt.xlabel('epoch')
    plt.ylabel('class error')
    plt.title('class error vs. epoch')

    plt.show()

# time against batch size
def time_graph(time_update, batch_sizes):
    plt.figure(2)
    plt.plot(range(len(batch_sizes)), time_update)
    plt.xticks(range(len(batch_sizes)), batch_sizes)
    plt.xlabel('batch size')
    plt.ylabel('time to update (ms)')
    plt.title('time to update vs. batch size')

    plt.show()

main()
