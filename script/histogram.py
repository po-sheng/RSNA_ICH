import numpy as np
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt

def build(img, name):
    
    cnt = Counter()

    for h in range(len(img)):
        for w in range(len(img[0])):
            cnt[int(img[h][w])] += 1
   
    x = []
    for idx in range(256):
        for count in range(cnt[idx]):
            x.append(idx)

    draw(x, name)
    return

def draw(x, name):
    matplotlib.use("Agg")

    plt.hist(x, 256)
    plt.title("histogram of "+name)
#     plt.xticks(np.arange(0, 256, 1))
    plt.xlabel("color value")
    plt.ylabel("count")

    plt.savefig(name+"_hist.png")
    
    return
