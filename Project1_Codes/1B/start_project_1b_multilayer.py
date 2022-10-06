#
# Project 1, starter code part b
#

import tensorflow as tf
import numpy as np
import pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as p
from sklearn import preprocessing
from sklearn import utils
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

NUM_FEATURES = 7
NUM_HIDDEN = 10

learning_rate = 0.01
epochs = 1000
batch_size = 8
#num_neuron = 30
seed = 10
np.random.seed(seed)

def init_bias(n = 1):
    return(tf.Variable(np.zeros(n), dtype=tf.float32))
        
def init_weights(n_in=1, n_out=1, logistic=True):
    W_values = np.asarray(np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)), size=(n_in, n_out)))
    if logistic == True:
        W_values *= 4
        return(tf.Variable(W_values, dtype=tf.float32))

def ffn_3(x, hidden1_units,numfeat):
        # Hidden 1
    with tf.name_scope('hidden1'):
        weights = init_weights(numfeat, hidden1_units)
        biases = init_bias(hidden1_units)
        #tf.cast(x, tf.float32)
        hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)

    with tf.name_scope('linear'):
        weights = init_weights(hidden1_units, 1)
        biases = init_bias(1)
        logits = tf.matmul(hidden1, weights) + biases

    return logits
        
def ffn_4(x, hidden1_units, hidden2_units):
        # Hidden 1
    with tf.name_scope('hidden1'):
        weights = init_weights(numfeat, hidden1_units)
        biases = init_bias(hidden1_units)
        #tf.cast(x, tf.float32)
        hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)
        hidden1 = dropout(hidden1)

    with tf.name_scope('hidden2'):
        weights = init_weights(hidden1_units, hidden2_units)
        biases = init_bias(hidden1_units)
        #tf.cast(x, tf.float32)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
        hidden2 = dropout(hidden2)

    with tf.name_scope('linear'):
        weights = init_weights(hidden2_units, 1)
        biases = init_bias(1)
        logits = tf.matmul(hidden2, weights) + biases

    cross_entropy = - nd.nansum(y * nd.log_softmax(logits), axis=0, exclude=True)
    for e in range(epochs):
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx).reshape((-1,784))
            label = label.as_in_context(ctx)
            label_one_hot = nd.one_hot(label, 10)
            with autograd.record():
                output = f(data, drop_prob=.5)
                loss = softmax_cross_entropy(output, label_one_hot)
            loss.backward()

            if i == 0:
                moving_loss = nd.mean(loss).asscalar()
            else:
                moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))
    

def ffn_5(x, hidden1_units, hidden2_units, hidden3_units):
        # Hidden 1
    with tf.name_scope('hidden1'):
        weights = init_weights(numfeat, hidden1_units)
        biases = init_bias(hidden1_units)
        #tf.cast(x, tf.float32)
        hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)

    with tf.name_scope('hidden2'):
        weights = init_weights(hidden1_units, hidden2_units)
        biases = init_bias(hidden1_units)
        #tf.cast(x, tf.float32)
        hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)

    with tf.name_scope('hidden3'):
        weights = init_weights(hidden2_units, hidden3_units)
        biases = init_bias(hidden1_units)
        #tf.cast(x, tf.float32)
        hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)

    with tf.name_scope('linear'):
        weights = init_weights(hidden3_units, 1)
        biases = init_bias(1)
        logits = tf.matmul(hidden1, weights) + biases

def train(index,numfeat):
        #read and divide data into test and train sets 
        admit_data = np.genfromtxt('admission_predict.csv', delimiter= ',')
        X_data, Y_data = admit_data[1:,index+1:numfeat+1], admit_data[1:,-1]
        Y_data = Y_data.reshape(Y_data.shape[0], 1)
        train_data=admit_data[1:281,:]
        test_data=admit_data[281:,:]
        

        idx = np.arange(X_data.shape[0])
        np.random.shuffle(idx)
        X_data, Y_data = X_data[idx], Y_data[idx]

        
        # experiment with small datasets
        trainX = X_data[:100]
        trainY = Y_data[:100]

        X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.3)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)


        trainX = (trainX- np.mean(trainX, axis=0))/ np.std(trainX, axis=0)
        predicted_output = []
        target_output= []

        # Create the model
        x = tf.placeholder(tf.float32, [None, numfeat])
        y_ = tf.placeholder(tf.float32, [None, 1])
        y = ffn_4(x,50,50,numfeat)

        # Build the graph for the deep net
        #weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, 1], stddev=1.0 / np.sqrt(NUM_FEATURES), dtype=tf.float32), name='weights')
        #biases = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases')
        #y = tf.matmul(x, weights) + biases



        #Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        loss = tf.reduce_mean(tf.square(y_ - y))
        train_op = optimizer.minimize(loss)



        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                train_acc = []
                test_acc=[]
                train_err = []
                test_err=[]
                for i in range(epochs):
                        train_op.run(feed_dict={x: trainX, y_: trainY})
                        err = loss.eval(feed_dict={x: trainX, y_: trainY})
                        train_err.append(err)

                        err1 = loss.eval(feed_dict={x: X_test, y_: y_test})
                        test_err.append(err1)

                        

                        if i % 100 == 0:
                                print('iter %d: train error %g'%(i, train_err[i]))
                                print('iter %d: test error %g'%(i, test_err[i]))

                        train_acc.append(accuracy.eval(feed_dict={x: x_tr, y_: y_tr}))
                        test_acc = accuracy.eval(feed_dict={x: x_te, y_: y_te})
                        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

                print('test acc:', test_acc)
                print('train_acc:', train_acc)
                
                paras = np.zeros(2)
                paras[0] = loss.eval(feed_dict={x: trainX, y_: trainY})
                paras[1] = loss.eval(feed_dict={x: X_test, y_: y_test})
                #print('train error', train_err)
                #print('train error', train_err)



        return train_err, test_err

def correlation_matrix():
     admit_data = np.genfromtxt('admission_predict.csv', delimiter= ',')
     X_data, Y_data = admit_data[1:,1:8], admit_data[1:,-1]
     x_train, x_test, y_train_, y_test_ = train_test_split(X_data, Y_data, test_size=0.3, random_state=42)
     Y_data = Y_data.reshape(Y_data.shape[0], 1)
     #scaler = preprocessing.StandardScaler()
     #x_train = scaler.fit_transform(x_train)
     #x_test = scaler.fit_transform(x_test)
     #y_train = y_train_.reshape(len(y_train_), )
     #y_test = y_test_.reshape(len(y_test_), )
     data = admit_data
     df = pd.DataFrame(data, columns = ['Serial No.','GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research','Chance of Admit'])
     df.corr()
     cors = df.corr()
     cors = df.drop("Serial No.", axis=1).corr() #to remove serial no. from the plot
     plt.matshow(cors)
     plt.yticks(range(cors.shape[1]), cors.columns, fontsize=7)
     plt.xticks(range(cors.shape[1]), cors.columns, fontsize=7, rotation=90)
     plt.colorbar()
     plt.show()

def Recursive():
     a = 7
     admit_data = np.genfromtxt('admission_predict.csv', delimiter= ',')
     data = admit_data
     names = ['Serial No.','GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research','Chance of Admit']
     df = pd.DataFrame(data, columns = ['Serial No.','GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research','Chance of Admit'])
     df = df.drop([0], axis=0)
     #print (df)
     array = df.values
     #print (array)
     X = array[:,1:8]
     print(X)
     Y = np.asarray(array[:,8])
     while (a >= 6):
         model = LinearRegression()
         rfe = RFE(model)
         fit = rfe.fit(X, Y)
         rank = fit.ranking_.tolist()
         index = rank.index(max(rank))
         df = df.drop(df.columns[[index]], axis=1)
         #print(df)
         numfeat=len(rank)
         main(index,numfeat)
  
         print(fit.ranking_)
         X=X[:,index+1:8]
         print (X)
         a -=1
         

     print("Num Features: %d"% fit.n_features_)
     print("Selected Features: %s"% fit.support_)
     print("Feature Ranking: %s"% fit.ranking_)


def main(index,numfeat):
        #paras=train()
        #paras = np.array(paras)
        batch_sizes = [8]

        train_err, test_err = train(index,numfeat)
                
        # plot learning curves
        plt.figure(1)
        plt.plot(range(epochs), train_err)
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Train Error')
        plt.show()

        plt.figure(2)
        plt.plot(range(epochs), test_err)
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Test Error')
        plt.show()
        

def dropout(X):
    x = tf.placeholder(tf.float32, [None, numfeat])

    with tf.Session() as sess:
        xdata = sess.run(x, feed_dict={x:X})
    keep_probability = 0.8
    x=np.array(X)
    mask = nd.random_uniform(0, 1.0, xdata.shape, ctx=xdata.context) < keep_probability
    if keep_probability > 0.0:
        scale = (1/keep_probability)
    else:
        scale = 0.0
    return mask * X * scale


'''
def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()
'''

#main()
Recursive()

