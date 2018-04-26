### 学習用のループ

import tensorflow as tf
import crawl
import model
#from model import *

TRAIN_NUM = 20000

#データセットの読み込み 
train_data = []
f = open("train_data.pkl", "rb")
train_data = pickle.load(f)

with tf.Graph().as_default() as graph:
    x = tf.placeholder(tf.float32, shape=[None, 2352])
    y_ = tf.placeholder(tf.float32, shape=[None, 257])
    keep_prob = tf.placeholder(tf.float32)

    #モデルの初期化
    model_obj = model.initialize(x, keep_prob)

    loss_val = loss(model_obj, y_)
    train_opt = training(loss_val)
    accuracy_val = accuracy(model_obj, y_)
    sess = tf.session()
    sess.run(tf.global_variables_initializer())
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("data", graph=sess.graph)

with tf.Session() as sess:
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, "./train_model/model.ckpt")
    #print("model restored")

    for i in range(TRAIN_NUM):
        batch = crawl.batch_train(train_data, 50)
        if i % 10 == 0:
            train_accuracy = sess.run(accuracy_val, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            saver.save(sess, "./train_model/model.ckpt")

        sess.run(train_opt, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        #TensorBoardに表示する値を追加
        summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_:batch[1], keep_prob: 1.0})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()

    batch = crawl.batch_train(train_data, 100)
    print("test accuracy %g" % sess.run(accuracy_val, feed_dict={
        x: batch[0], y_: batch[1], keep_prob: 1.0}))

    saver.save(sess, "./train_model/model.ckpt")
    print("model saved")