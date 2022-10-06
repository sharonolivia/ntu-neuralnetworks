#
# Project 1, starter code part b
#

import tensorflow as tf
import numpy as np
import pylab as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras



tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

NUM_FEATURES = 7
NUM_HIDDEN = 10

learning_rate = 0.01
epochs = 1000
batch_size = 8
#num_neuron = 30
seed = 10
np.random.seed(seed)
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

def init_bias(n = 1):
    return(tf.Variable(np.zeros(n), dtype=tf.float32))
       
def init_weights(n_in=1, n_out=1, logistic=True):
    W_values = np.asarray(np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)), size=(n_in, n_out)))
    if logistic == True:
        W_values *= 4
        return(tf.Variable(W_values, dtype=tf.float32))

def ffn(x, hidden1_units):
        # Hidden 1
    with tf.name_scope('hidden1'):
        weights = init_weights(NUM_FEATURES, hidden1_units)
        biases = init_bias(hidden1_units)
        #tf.cast(x, tf.float32)
        hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)

    with tf.name_scope('linear'):
        weights = init_weights(hidden1_units, 1)
        biases = init_bias(1)
        logits = tf.matmul(hidden1, weights) + biases

    return logits
       


def train():
        #read and divide data into test and train sets
        admit_data = np.genfromtxt('admission_predict.csv', delimiter= ',')
        X_data, Y_data = admit_data[1:,1:8], admit_data[1:,-1]
        Y_data = Y_data.reshape(Y_data.shape[0], 1)

        idx = np.arange(X_data.shape[0])
        np.random.shuffle(idx)
        X_data, Y_data = X_data[idx], Y_data[idx]

       
        # experiment with small datasets
        trainX = X_data[:100]
        trainY = Y_data[:100]

        X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.3)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)


        X_train = (X_train- np.mean(X_train, axis=0))/ np.std(X_train, axis=0)
        predicted_output = []
        target_output= []

        # Create the model
        x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
        y_ = tf.placeholder(tf.float32, [None, 1])
        y = ffn(x, 10)
        #y = scale(y, np.min(y, axis=0), np.max(y, axis=0))

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
                train_err = []
                test_err=[]
                for i in range(epochs):
                    sess.run(fetches=[train_op],feed_dict={x: X_train, y_: y_train})
                    #train_op.run(feed_dict={x: trainX, y_: y_train})
                    err = loss.eval(feed_dict={x: X_train, y_: y_train})
                    train_err.append(err)

                    err1 = loss.eval(feed_dict={x: X_test, y_: y_test})
                    test_err.append(err1)

                    if i % 100 == 0:
                        print('iter %d: train error %g'%(i, train_err[i]))
                        print('iter %d: test error %g'%(i, test_err[i]))

                expected_scores=sess.run(fetches=y, feed_dict={x: X_train})
                expected_scores = scale(expected_scores, np.min(expected_scores, axis=0), np.max(expected_scores, axis=0))
                print("Expected Scores : ", expected_scores)
               
                plotdataT = sess.run(fetches=y, feed_dict={y:y_train[:50]})
                print('plot target:',plotdataT)
                plotdataP = sess.run(fetches=y, feed_dict={y:expected_scores[:50]})
                print('plot pred:', plotdataP)

                paras = np.zeros(2)
                paras[0] = loss.eval(feed_dict={x: X_train, y_: y_train})
                paras[1] = loss.eval(feed_dict={x: X_test, y_: y_test})

        return train_err, test_err, plotdataT, plotdataP


def main():
        #paras=train()
        #paras = np.array(paras)
        batch_sizes = [8]

        train_err, test_err, plotdataT, plotdataP = train()
               
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

        plotdataT=np.array(plotdataT)
        plotdataP=np.array(plotdataP)
#        plot_target=plotdata[0]
#        plot_predicted=plotdata[1]

        plt.figure(figsize=(1,1))
        plot1=plt.scatter(plotdataT, plotdataT, marker='^', color='red')
        plot2=plt.scatter(plotdataP, plotdataP, marker='o', color='blue')
        plt.legend((plot1,plot2),('Predicted','Actual'))
        plt.title('target vs pred')
        plt.xlabel('target')
        plt.ylabel('pred')
        plt.show()

main()
