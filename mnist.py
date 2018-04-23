### 全結合ニューラルネットワークによるMNIST画像認識
### このプログラムで構築しているネットワークは多層ではないので、深層学習ではありません

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNISTデータの読み込み(初回起動時はネット環境必須)

# ------------MNISTデータの概要----------------
# 60000点の訓練データと10000点のテストデータが存在
# 画像データのサイズは28*28=784
# この画像データに対応するラベル(0から9)がひとつずつ存在
# 例: 画像が4の場合のラベル　[0,0,0,0,1,0,0,0,0,0]
# このようにひとつだけ1で残りが0となっているようなベクトルをone hotベクトルという

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# モデル定義(ここでネットワークの定義を行う)

# 訓練画像を入れる変数
# Noneは任意の次元の長さにすることを表す(今回の場合、任意の数だけ訓練画像を入れられる)
x = tf.placeholder(tf.float32, [None, 784])

# 重み
# 列はラベルの数だけ
# 0で初期化
W = tf.Variable(tf.zeros([784, 10]))

# バイアス
# 列はラベルの数だけ
# 0で初期化
b = tf.Variable(tf.zeros([10]))

# 出力用の変数
# matmulで乗算を行う
# Wx + bの結果にソフトマックス関数を適用して出力とする
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 正解データのラベル
y_ = tf.placeholder(tf.float32, [None, 10])

# クロスエントロピー関数
# 正解データy_と出力されたyの値を比較して誤差関数を求める
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# 勾配降下法を用いてクロスエントロピー関数が最小になるようにパラメータを最適化する
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# 先ほど作成した変数を初期化する
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# 学習ループ
# 1000回の学習を実行する
for i in range(1000):
    # 訓練データからランダムに100個選びバッチデータを作成する
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 学習を実行
    # feed_dictでplaceholderに値を入力できる
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 正解数を求める
# 正解だと予測したラベルと、正解データを比較して同じであればTrueが返る
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# 正解率を求める
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 学習結果を表示する
print("正解率: ", end="")
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))