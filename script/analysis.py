import os
import sys
import cv2
import shutil
import histogram
from tqdm import tqdm
import pydicom
from multiprocessing import Pool
import numpy as np
from sklearn.model_selection import train_test_split

def mp(path_idx_save):
        
        path = path_idx_save[0]
        idx = path_idx_save[1]
        savePath = path_idx_save[2]
        
        maxNum = -10000000
        minNum = 10000000

        # Show the attributes of ds 
        ds = pydicom.read_file(path)
        # print(ds.dir("pat"))

        # Binary file for raw image
        pixel_bytes = ds.PixelData
        # print(type(pixel_bytes))              # byte type

        # Get image matrix
        img = ds.pixel_array                    # ndarray
    
        minNum = img.min()
        maxNum = img.max()

        img = (((img - minNum) / (maxNum - minNum)) * 255)

        label = path.split("/")[-2]
#         cv2.imwrite(savePath+label+"/"+label+"_"+str(idx)+".png", img)
        
        # epidural, subdural, subarachnoid, intraventricular, healthy, intraparenchymal
        if idx == 259:                  
            histogram.build(img, label+"_"+str(idx))

        return maxNum, minNum

def dataLoader(path: str, savePath: str):
    '''
        Read all the image in "Traindata" and split it to train and val

        - Inputs -
        ----------------------------
        path: str
            the path to dataset    
        savePath: str
            path to save the normalized image
        ============================
        
        - Outputs -
        ----------------------------
        train and val set
    '''
    np.set_printoptions(threshold=sys.maxsize)
    
    train = {}
    test = {}

    data = []
    labels = []
        
    maxNum = -10000000
    minNum = 10000000
    print(os.listdir(path))    
    for label in os.listdir(path):
        print(label)

        if os.path.isdir(savePath+label): 
            shutil.rmtree(savePath+label)
        os.mkdir(savePath+label)
        
        images = os.listdir(path + label + "/")
        
        for idx in range(len(images)):
            images[idx] = (path + label + "/" + images[idx], idx, savePath)

        with Pool(50) as p:
            max_min_tuple = p.map(mp, images)

        maxNumList = [tup[0] for tup in max_min_tuple]
        minNumList = [tup[1] for tup in max_min_tuple]

        if maxNum < max(maxNumList):
            maxNum = max(maxNumList)
        if minNum > min(minNumList):
            minNum = min(minNumList)

    print(maxNum, minNum)

    return 0, 0


if __name__ == "__main__":
    
    # Path to dataset
    path = "../dataset/raw/"
    savePath = "../dataset/norm/"
    
    train, val = dataLoader(path, savePath)
