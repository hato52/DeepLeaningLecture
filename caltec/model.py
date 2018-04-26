### ネットワークのモデルを記述

import tensorflow as tf

INPUT_SIZE = 2352
OUTPUT_SIZE= 257
CHANNEL_SIZE = 3

#モデルのイニシャライズ
def initialize(img_placeholder, keep_prob):

    #正規分布で重みを初期化
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    #0.1でバイアスを初期化
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    #畳み込み処理
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

    #プーリング処理
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    #train_data = data.read_data()
    #sess = tf.InteractiveSession()

    #入出力の大きさ定義
    #x = tf.placeholder(tf.float32, shape=[None, 2352])
    #y_ = tf.placeholder(tf.float32, shape=[None, 257])

    x_image = tf.reshape(img_placeholder, [-1, 28, 28, 3])

    ### 1層目 畳み込み層1
    with tf.variable_scope("conv1") as scope:
        #重みとバイアス
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        #活性化関数
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    ### 2層目 プーリング層1
    with tf.variable_scope("pool1") as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    ### 3層目 畳み込み層2
    with tf.variable_scope("conv2") as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    ### 4層目 プーリング層2
    with tf.variable_scope("pool2") as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    ### 5層目 全結合層
    with tf.variable_scope("fc") as scope:
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        #ドロップアウトの適用
        #keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ### 6層目 softmax関数による正規化
    with tf.variable_scope("softmax") as scope:
        W_fc2 = weight_variable([1024, 257])
        b_fc2 = bias_variable([257])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        return y_conv

#誤差関数の定義
def loss(y_conv, y_):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

#オプティマイザの定義
def training(cross_entropy):    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    return train_step

#正解率の計算
def accuracy(y_conv, y_):    
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, -1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy