#
# Project 2, starter code Part a
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
    

def main(numMaps1, numMaps2, counter):

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
    train_step_MO = tf.train.MomentumOptimizer(learning_rate, 0.1).minimize(cross_entropy)
    train_step_RMS = tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy)
    train_step_AO = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    print('NO. OF FEATURE MAPS 1:',numMaps1)
    print('NO. OF FEATURE MAPS 2:',numMaps2)

    N = len(trainX)
    idx = np.arange(N)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        test_acc = []
        training_cost = []
        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})

            _, loss_ = sess.run([train_step, loss], {x: trainX, y_: trainY})
            print('epoch', e, 'entropy', loss_)
                
            training_cost.append(loss.eval(feed_dict={x: trainX, y_: trainY}))
            print('epoch %d: training cost %g'%(e, training_cost[e]))

            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))
            print('epoch %d: test accuracy %g'%(e, test_acc[e]))


        if (numMaps1==50 and numMaps2==60):

            ind1 = np.random.randint(low=0, high=len(testX))
            ind2 = np.random.randint(low=0, high=len(testX))

            print("pattern 1 index:", ind1)
            print("pattern 2 index:", ind2)

            pattern1 = testX[ind1,:]
            pattern2 = testX[ind2,:]

            conv1_, pool1_, conv2_, pool2_ = sess.run([conv_1, pool_1, conv_2, pool_2],{x: np.expand_dims(pattern1,0)})
            conv12_, pool12_, conv22_, pool22_ = sess.run([conv_1, pool_1, conv_2, pool_2],{x: np.expand_dims(pattern2,0)})

            
            # PATTERN 1
            plt.figure('pattern 1')
            plt.gray()
            plt.imshow(pattern1.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0))
                
            # convolution layer 1
            plt.figure("conv1_maps")
            plt.gray()
            conv1_ = np.array(conv1_)
            for i in range(numMaps1):
                plt.subplot(5, 10, i+1); plt.axis('off'); plt.imshow(conv1_[0,:,:,i])
            plt.savefig('./figures_partA/Pattern1_ConvLayer1'+'_'+str(numMaps1)+'_'+str(numMaps2)+'.png')

            # pool layer 1
            plt.figure("pool1_maps")
            plt.gray()
            pool1_ = np.array(pool1_)

            for i in range(numMaps1):
                plt.subplot(5, 10, i+1); plt.axis('off'); plt.imshow(pool1_[0,:,:,i])
            plt.savefig('./figures_partA/Pattern1_PoolingLayer1'+'_'+str(numMaps1)+'_'+str(numMaps2)+'.png')
            
            # convolution layer 2
            plt.figure("conv2_maps")
            plt.gray()
            conv2_ = np.array(conv2_)
            for i in range(numMaps2):
                plt.subplot(6, 10, i+1); plt.axis('off'); plt.imshow(conv2_[0,:,:,i])
            plt.savefig('./figures_partA/Pattern1_ConvLayer2'+'_'+str(numMaps1)+'_'+str(numMaps2)+'.png')

            # pool layer 2
            plt.figure("pool2_maps")
            plt.gray()
            pool2_ = np.array(pool2_)
            for i in range(numMaps2):
                plt.subplot(6, 10, i+1); plt.axis('off'); plt.imshow(pool2_[0,:,:,i])
            plt.savefig('./figures_partA/Pattern1_PoolingLayer2'+'_'+str(numMaps1)+'_'+str(numMaps2)+'.png')



            # PATTERN 2
            plt.figure('pattern 2')
            plt.gray()
            plt.imshow(pattern2.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0))

            # convolution layer 1
            plt.figure("pattern2_conv1")
            plt.gray()
            conv1_ = np.array(conv12_)
            for i in range(numMaps1):
                plt.subplot(5, 10, i+1); plt.axis('off'); plt.imshow(conv1_[0,:,:,i])
            plt.savefig('./figures_partA/Pattern2_ConvLayer1'+'_'+str(numMaps1)+'_'+str(numMaps2)+'.png')

            # pool layer 1
            plt.figure("pattern2_pool1")
            plt.gray()
            pool1_ = np.array(pool12_)

            for i in range(numMaps1):
                plt.subplot(5, 10, i+1); plt.axis('off'); plt.imshow(pool1_[0,:,:,i])
            plt.savefig('./figures_partA/Pattern2_PoolingLayer1'+'_'+str(numMaps1)+'_'+str(numMaps2)+'.png')
            
            # convolution layer 2
            plt.figure("pattern2_conv2")
            plt.gray()
            conv2_ = np.array(conv22_)
            for i in range(numMaps2):
                plt.subplot(6, 10, i+1); plt.axis('off'); plt.imshow(conv2_[0,:,:,i])
            plt.savefig('./figures_partA/Pattern2_ConvLayer2'+'_'+str(numMaps1)+'_'+str(numMaps2)+'.png')

            # pool layer 2
            plt.figure("pattern2_pool2")
            plt.gray()
            pool2_ = np.array(pool22_)
            for i in range(numMaps2):
                plt.subplot(6, 10, i+1); plt.axis('off'); plt.imshow(pool2_[0,:,:,i])
            plt.savefig('./figures_partA/Pattern2_PoolingLayer2'+'_'+str(numMaps1)+'_'+str(numMaps2)+'.png')

            #plt.show()

    # ACCURACY GRAPH
    plt.figure(counter)
    acc_label='gradient descent'+'_'+str(numMaps1)+'_'+str(numMaps2)
    plt.plot(np.arange(epochs), test_acc, label=acc_label)
    plt.xlabel('epochs')
    plt.ylabel('test accuracy')
    plt.legend(loc='lower right')
    plt.savefig('./figures_partA/test_acc'+'_'+str(numMaps1)+'.png')
    #plt.show()

    # COST GRAPH
    plt.figure(counter+12)
    cost_label='gradient descent'+'_'+str(numMaps1)+'_'+str(numMaps2)
    plt.plot(np.arange(epochs), training_cost, label=cost_label)
    plt.xlabel('epochs')
    plt.ylabel('training cost')
    plt.legend(loc='lower right')
    plt.savefig('./figures_partA/training_cost'+'_'+str(numMaps1)+'.png')
    #plt.show()
    
    plt.figure()
    plt.gray()
    X_show = X.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)
    plt.axis('off')
    plt.imshow(X_show)
    plt.savefig('./figures_partA/p1b_2'+'_'+str(numMaps1)+'_'+str(numMaps2)+'.png')


def run():
    num_maps1 = [10,20,30,40,50,60,70,80,90,100]
    num_maps2 = [10,20,30,40,50,60,70,80,90,100]
    counter=1
    for value1 in num_maps1:
        for value2 in num_maps2:
            main(value1,value2,counter)
        counter += 1


if __name__ == '__main__':
    run()
