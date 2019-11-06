import tensorflow as tf
import random as rnd
import model as model
import tools as tools
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

rawData_len = 600               # length of data

step_rate = 1e-4                # learning rate
class_n = 7                     # number of the softmax label
mode = 0
                       # 0:train   1:predict
if mode == 0:
    batch_total = 80            # total number of batch
    max_steps = 10000        # the max epoch
    batch_size = 7              # the max epoch
    inner_loop = 10             # number of the inner loop
else:
    batch_total = 20
    max_steps = batch_total
    batch_size = 7
    inner_loop = 1


pltLoss = []                    # loss of the neural network for plot
pltError = []                   # accuracy for plot
b = []                          # index of the axis for plot
n = 0                           # counter

training = tf.placeholder(tf.bool)                                  # parameter normalization
x = tf.placeholder(tf.float32, shape=[batch_size, rawData_len])     # raw data input of model
y = tf.placeholder(tf.int32, shape=[batch_size, class_n])           # label of
x_len = tf.placeholder(tf.int32, shape=[batch_size, ])

logits, cnn_feature = model.inference(x, training, class_n)         # neural network output
loss = model.loss(logits, y)                                        # loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=step_rate).minimize(loss)   # optimizer
error = model.prediction(logits, y)                                 # accuracy calculate

init = tf.global_variables_initializer()                            # initial all variables of model

saver = tf.train.Saver()                                            # save and load models

with tf.Session() as sess:
    if mode == 0:                                   # train
        sess.run(init)
    else:                                           # predict
      saver.restore(sess, './model/model.ckpt')

    for num_iter in range(max_steps):
        if mode == 0:                               # train mode
            line_index = rnd.randint(0, batch_total - 1) * class_n  # random reading index of the load data
            data_ecg, data_ecg_len, data_label = tools.load_data_set(mode, line_index, batch_size, class_n)   # load data

            for loop in range(inner_loop):
                val_opt = sess.run(optimizer, feed_dict={x: data_ecg, y: data_label, x_len: data_ecg_len, training: True})

            if num_iter % 10 == 0:
                v_loss = sess.run(loss, feed_dict={x: data_ecg, y: data_label,  x_len: data_ecg_len, training: True})
                print('loop %d of %d     err = %f     line = %d' % (num_iter, max_steps, v_loss, line_index))
                pltLoss.append(v_loss)
                b.append(n)
                n = n + 1
        else:                                       # predict mode
            line_index = num_iter * class_n         # linear reading index of the load data
            data_ecg, data_ecg_len, data_label = tools.load_data_set(mode, line_index, batch_size, class_n)

            accuracy_single, accuracy_mean = sess.run(error, feed_dict={x: data_ecg, y: data_label,  x_len: data_ecg_len, training: True})
            print('accuracy mean : %f' % accuracy_mean)

    if mode == 0:                                   # train
        plt.plot(b, pltLoss)
        plt.show()
        saver.save(sess, './model/model.ckpt')   # save model

print('Finished!')
