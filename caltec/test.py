### テスト用
 
import pickle
import cv2
import tensorflow as tf
import crawl
import model
import numpy as np
#from model import *

#テスト画像の入力
img_path = input("画像のパスを入力して下さい >")
img = cv2.imread(img_path)
img = cv2.resize(img, (28, 28))
input_img = img.flatten().astype(np.float32)/255.0

#データセットの読み込み
train_data = []
f = open("train_data.pkl", "rb")
train_data = pickle.load(f)

x = tf.placeholder(tf.float32, shape=[None, 2352])
y_ = tf.placeholder(tf.float32, shape=[None, 257])
keep_prob = tf.placeholder(tf.float32)

model_obj = model.initialize(x, keep_prob)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./train_model/model.ckpt")
    print("model restored")

    #batch = crawl.batch_train(train_data, 1)
    #print("test accuracy %g" % accuracy.eval(feed_dict={
    #    x: batch[0], y_: batch[1], keep_prob: 1.0}))

    result = model_obj.eval(feed_dict={x: [input_img], keep_prob: 1.0})
    result_list = np.argsort(result[0])[::-1]
    result_list = result_list[:5]
    i = 0
    for num in result_list:
        i += 1
        print("候補" + str(i) + "番目: ", end="")
        print(train_data[2][num])

    #result = np.argmax(model_obj.eval(feed_dict={x: [input_img], keep_prob: 1.0})[0])
    #print(train_data[2][result])