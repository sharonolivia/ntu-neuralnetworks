import numpy as np
import pandas
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
from timeit import default_timer as timer

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 20

batch_size = 128
epochs = 100
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def rnn_model(x):

  word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

  word_list = tf.unstack(word_vectors, axis=1)

  cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
  _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return logits, word_list

def data_read_words():
  
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
  y_train = y_train.values
  y_test = y_test.values
  
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH)

  x_transform_train = vocab_processor.fit_transform(x_train)
  x_transform_test = vocab_processor.transform(x_test)

  x_train = np.array(list(x_transform_train))
  x_test = np.array(list(x_transform_test))

  no_words = len(vocab_processor.vocabulary_)
  print('Total words: %d' % no_words)

  return x_train, y_train, x_test, y_test, no_words

def configure_statistics(logits, y_):
  correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)
  classification_errors = tf.count_nonzero(tf.not_equal(tf.argmax(logits, 1), tf.argmax(y_, 1)))
  return correct_prediction, accuracy, classification_errors


def main():
  global n_words

  x_train, y_train, x_test, y_test, n_words = data_read_words()

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  logits, word_list = rnn_model(x)
  correct_prediction, accuracy, classification_errors = configure_statistics(logits, tf.one_hot(y_, MAX_LABEL))


  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
  minimizer = tf.train.AdamOptimizer(lr)

  # Gradient clipping
  grads_and_vars = minimizer.compute_gradients(entropy)
  grad_clipping = tf.constant(2.0, name="grad_clipping")
  clipped_grads_and_vars = []
  for grad, var in grads_and_vars:
      clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
      clipped_grads_and_vars.append((clipped_grad, var))
  # Gradient updates
  train_op = minimizer.apply_gradients(clipped_grads_and_vars)
  N = len(x_train)
  indexes = np.arange(N)

  with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    # training
    test_acc_pts = []
    training_cost_pts = []
    training_acc_pts = []
    epoch_times = []

    randomizedX, randomizedY = x_train,y_train
    testX,testY = x_test,y_test

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
        
    np.savetxt('./Q6c_word_test_accs.txt',np_test_accs)
    np.savetxt('./Q6c_word_time.txt',np_times)

    print('Time taken', total_end - total_start)
    
    plt.figure(1)
    plt.plot(range(epochs), test_acc_pts)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Accuracy against Test Data')
    plt.savefig('./Accuracy_vs_Epochs_Q6c_word.png')

    plt.figure(2)
    plt.plot(range(epochs), training_cost_pts)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Training Cost')
    plt.savefig('./TrainingCost_vs_epochs_Q6c_word.png')

    plt.figure(3)
    plt.plot(range(epochs), training_acc_pts)
    plt.plot(range(epochs), test_acc_pts)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Accuracy ')
    plt.legend(["Training Accuracy","Test Accuracy"])
    plt.savefig('./Train Acc vs Test Acc_Q6c_word.png')

    plt.show()

    sess.close()
  
  
if __name__ == '__main__':
  main()
