import numpy as np
import pandas
import tensorflow as tf
import csv
from timeit import default_timer as timer
import matplotlib.pyplot as plt

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 20

epochs = 100
lr = 0.01
batch_size = 128

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def rnn_model(x):

  input_layer = tf.reshape(tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256]) 
  char_list = tf.unstack(input_layer, axis=1)

  cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
  _, encoding = tf.nn.static_rnn(cell, char_list, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return logits, char_list

def data_read_chars():
  
  x_train, y_train, x_test, y_test = [], [], [], []
  
  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[2])
      y_train.append(int(row[0]))

  with open("test_medium.csv", encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[2])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  
  
  char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(char_processor.fit_transform(x_train)))
  x_test = np.array(list(char_processor.transform(x_test)))
  y_train = y_train.values
  y_test = y_test.values
  

  return x_train, y_train, x_test, y_test

def configure_statistics(logits, y_):
  correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)
  classification_errors = tf.count_nonzero(tf.not_equal(tf.argmax(logits, 1), tf.argmax(y_, 1)))
  return correct_prediction, accuracy, classification_errors

def main():
  
  x_train, y_train, x_test, y_test = data_read_chars()

  print(len(x_train))
  print(len(x_test))

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  logits, char_list = rnn_model(x)
  correct_prediction, accuracy, classification_errors = configure_statistics(logits, tf.one_hot(y_, MAX_LABEL))

  # Optimizer
  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
  train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

  N = len(x_train)
  indexes = np.arange(N)

  with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    # training
    test_acc_pts = []
    training_acc_pts = []
    training_cost_pts = []
    epoch_times = []

    randomizedX, randomizedY = x_train,y_train

    total_start = timer()

    for e in range(epochs):

        np.random.shuffle(indexes)            
        randomizedX, randomizedY = randomizedX[indexes], randomizedY[indexes]

        experiment_start = timer()
        
        for start, end in zip(range(0, N+1, batch_size), range(batch_size, N+1, batch_size)):
            sess.run([train_op], {x: randomizedX[start:end], y_: randomizedY[start:end]})
            
        experiment_end = timer()

        #upon completing an epoch of training, collect required stats
        loss_pt = entropy.eval(feed_dict={x: randomizedX, y_: randomizedY})
        training_cost_pts.append(loss_pt)
        test_acc_pt = accuracy.eval(feed_dict={x: x_test, y_: y_test})
        test_acc_pts.append(test_acc_pt)
        training_acc_pt = accuracy.eval(feed_dict={x: x_train, y_: y_train})
        training_acc_pts.append(training_acc_pt)
        epoch_times.append(experiment_end-experiment_start)
                

        
        print('epoch', e, 'entropy', loss_pt, 'time', experiment_end - experiment_start)

    total_end = timer()

    np_test_accs = np.array(test_acc_pts)
    np_test_accs = np.expand_dims(np_test_accs,axis=0)
    np_times = np.expand_dims((total_end - total_start, np.mean(epoch_times)),axis=0)
        
    np.savetxt('./Q6a_char_vanilla_test_accs.txt',np_test_accs)
    np.savetxt('./Q6a_char_vanilla_time.txt',np_times)

    print('Time taken', total_end - total_start)

    plt.figure(1)
    plt.plot(range(epochs), test_acc_pts)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Accuracy against Test Data')
    plt.savefig('./Accuracy_vs_Epochs_Q6a_char_vanilla.png')

    plt.figure(2)
    plt.plot(range(epochs), training_cost_pts)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Training Cost')
    plt.savefig('./TrainingCost_vs_epochs_Q6a_char_vanilla.png')

    plt.figure(3)
    plt.plot(range(epochs), training_acc_pts)
    plt.plot(range(epochs), test_acc_pts)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Accuracy ')
    plt.legend(["Training Accuracy","Test Accuracy"])
    plt.savefig('./Train acc vs Test Acc_Q6a_char_vanilla.png')
    
    plt.show()

    sess.close()


if __name__ == '__main__':
  main()
