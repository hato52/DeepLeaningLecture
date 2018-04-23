# ランダムに生成された5×5の白黒画像を表示してください
# ヒント1 配列は1次元ベクトルで表します
# ヒント2 np.random.randint() で整数の乱数を生成できます
# ヒント3 np.append() でリストの後ろに要素を追加できます

import numpy as np
import matplotlib.pyplot as plt
import random

# 画像をランダムで生成
def generateImg():
    #---この関数の処理を記述---
    vec = np.array([])

    for i in range(25):
        rndnum = random.randint(0,1)
        vec = np.append(vec, rndnum)

    return vec
    #------------------------

# 画像として出力
def showImg(imgVector):
    for i in range(25):
        if imgVector[i] == 0:
            imgVector[i] = 255
        elif imgVector[i] == 1:
            imgVector[i] = 0
        
    imgVector = np.reshape(imgVector, (5, 5))
    plt.imshow(imgVector, cmap="gray")
    plt.show()

# メイン処理
def main():
    img = generateImg()
    showImg(img)

if __name__ == "__main__":
    main()    
