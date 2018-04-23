### 畳み込みニューラルネットワークによるMNIST画像認識
### ここでは3層のニューラルネットワークを用いた深層学習を行います

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNISTデータの読み込み
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 正規分布で重みを初期化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 0.1でバイアスを初期化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 畳み込み処理
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

# プーリング処理
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

sess = tf.InteractiveSession()

# 訓練データ用のテンソルと正解データ用のテンソルを用意
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 28*28の画像にリシェイプ
x_image = tf.reshape(x, [-1, 28, 28, 1])

# --------------------1層目 畳み込み層1-------------------------

# 畳み込み層のフィルタの重みを設定(パッチサイズ縦、パッチサイズ横、入力チャネル数、出力チャネル数)
W_conv1 = weight_variable([5, 5, 1, 32])

# 畳み込み層のバイアス
b_conv1 = bias_variable([32])

# 活性化関数Reluを出力に適用
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# -------------------------------------------------------------


# --------------------2層目 プーリング層1-----------------------

# 2*2のマックスプーリング層を構築
h_pool1 = max_pool_2x2(h_conv1)

# -------------------------------------------------------------


# --------------------3層目 畳み込み層2-------------------------

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# -------------------------------------------------------------


# --------------------4層目 プーリング層2-----------------------

h_pool2 = max_pool_2x2(h_conv2)

# -------------------------------------------------------------


# --------------------5層目 全結合層----------------------------

# 現在の画像サイズは7*7まで減少
# 全結合層にするために1階テンソルに変形
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#ドロップアウトの適用(過学習を防ぐため)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# -------------------------------------------------------------


# --------------------6層目 出力層----------------------------

# ソフトマックス関数を適用して出力を正規化
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# -------------------------------------------------------------

# 各種評価系の関数を用意

# 誤差関数(クロスエントロピー)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# 勾配降下法の学習係数の指定(Adam)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 正解数
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, -1))

# 正解率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 学習ループ
# 学習回数をむやみに増やしすぎると、とんでもない時間がかかるので注意
# 2万回実行する場合、それなりのスペックでも30分はかかります

# 学習結果を保存するためのセーバーを用意
saver = tf.train.Saver()
# セッションを実行
sess.run(tf.global_variables_initializer())

for i in range(20000):
    # ミニバッチ学習
    batch = mnist.train.next_batch(50)

    # 100回ごとに途中経過を表示
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        # 途中結果を保存
        saver.save(sess, "./train_model/model.ckpt")

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# テストデータを用いて最終的な精度を表示
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))