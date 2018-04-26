### フォルダをクロールして学習データを出力

import os
import pickle
import numpy as np
import cv2

NUM_CLASSES = 257
IMG_SIZE = 28
IMG_PATH= "./caltec256/"

#画像のあるディレクトリ
train_img_dirs = []

#データセットを読み込んで返す
def read_dataset():
    print("Start Read DataSet...")
    #学習データの画像とラベル(出力するのはこれ)
    train_image = []
    train_label = []

    #ディレクトリのリストを作成
    print("Create DirectoryList")
    for fname in os.listdir(IMG_PATH):
        train_img_dirs.append(fname)

    #データのリストを作成
    print("Create DataList")
    for i, dir_name in enumerate(train_img_dirs):
        print("Read %s" % dir_name)
        files = os.listdir(IMG_PATH + dir_name)

        for file_name in files:
            #画像データを一次元にしてリストに追加
            try:
                img = cv2.imread(IMG_PATH + dir_name + "/" + file_name)
                height, width, ch = img.shape
            except:
                print(IMG_PATH + dir_name + "/" + file_name)

            #print("height: " + str(height) + " width: " + str(width))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.flatten().astype(np.float32)/255.0
            train_image.append(img)

            #one_hot_vectorをラベルとしてリストに追加
            tmp = np.zeros(NUM_CLASSES)
            tmp[i] = 1
            train_label.append(tmp)

    #numpy配列に変換
    train_image = np.asarray(train_image)
    train_label = np.asarray(train_label)

    train_data = []
    train_data.append(train_image)
    train_data.append(train_label)
    train_data.append(train_img_dirs)

    f = open("train_data.pkl", "wb")
    pickle.dump(train_data, f)
    f.close

    print("Done!!")
    return train_image, train_label


#バッチ学習を行うためのデータを返す
def batch_train(train_data, batch_size):

    train_image = train_data[0]
    train_label = train_data[1]
    #テストデータのリストをシャッフルして、バッチサイズだけ取得
    tmp = list(zip(train_image, train_label))
    np.random.shuffle(tmp)

    batch_list = tmp[:batch_size]
    batch_image, batch_label = zip(*batch_list)

    return np.asarray(batch_image), np.asarray(batch_label) 

if __name__ == "__main__":
    read_dataset()    