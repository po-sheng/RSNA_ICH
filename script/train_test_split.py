import os
import shutil
import glob
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

def split(readPath, savePath):
    
    # Create a new data split
    if os.path.isdir(savePath):
        shutil.rmtree(savePath)
    os.mkdir(savePath)
    os.mkdir(savePath+"train")
    os.mkdir(savePath+"val")

    labels = os.listdir(readPath)
    for label in labels:

        img = os.listdir(readPath+label)
        img_label = [label for _ in range(len(img))]

        # Split images
        x_train, x_val, _, _ = train_test_split(img, img_label, test_size=0.1, random_state=43, shuffle=True)

        # Move the file to corresponding directory
        count = 0
        for idx in range(len(x_train)):
            shutil.copy(readPath+label+"/"+x_train[idx], savePath+"train/"+label+"_"+str(count)+"."+x_train[idx].split(".")[-1])
            count += 1
        for idx in range(len(x_val)):
            shutil.copy(readPath+label+"/"+x_val[idx], savePath+"val/"+label+"_"+str(count)+"."+x_val[idx].split(".")[-1])
            count += 1


if __name__ == "__main__":
    
    readPath = "../dataset/raw/"
    savePath = "../dataset/split/"
    
    split(readPath, savePath)

