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

def Dicom_to_Image(Path):
    DCM_Img = pydicom.read_file(Path)

    #print(DCM_Img)

    #print(type(DCM_Img.get(0x00281050).value))

    rows = DCM_Img.get(0x00280010).value #Get number of rows from tag (0028, 0010)
    cols = DCM_Img.get(0x00280011).value #Get number of cols from tag (0028, 0011)

    if isinstance(DCM_Img.get(0x00281050).value, pydicom.multival.MultiValue):
        Window_Center = int(DCM_Img.get(0x00281050).value[0])
    else:
        Window_Center = int(DCM_Img.get(0x00281050).value) #Get window center from tag (0028, 1050)

    if isinstance(DCM_Img.get(0x00281051).value, pydicom.multival.MultiValue):
        Window_Width = int(DCM_Img.get(0x00281051).value[0])
    else:
        Window_Width = int(DCM_Img.get(0x00281051).value) #Get window width from tag (0028, 1051)

    Window_Max = int(Window_Center + Window_Width / 2)
    Window_Min = int(Window_Center - Window_Width / 2)
    
    if (DCM_Img.get(0x00281052) is None):
        Rescale_Intercept = 0
    else:
        Rescale_Intercept = int(DCM_Img.get(0x00281052).value)

    if (DCM_Img.get(0x00281053) is None):
        Rescale_Slope = 1
    else:
        Rescale_Slope = int(DCM_Img.get(0x00281053).value)

    New_Img = np.zeros((rows, cols), np.uint8)
    Pixels = DCM_Img.pixel_array

    for i in range(0, rows):
        for j in range(0, cols):
            Pix_Val = Pixels[i][j]
            Rescale_Pix_Val = Pix_Val * Rescale_Slope + Rescale_Intercept

            if (Rescale_Pix_Val > Window_Max): #if intensity is greater than max window
                New_Img[i][j] = 255
            elif (Rescale_Pix_Val < Window_Min): #if intensity is less than min window
                New_Img[i][j] = 0
            else:
                New_Img[i][j] = int(((Rescale_Pix_Val - Window_Min) / (Window_Max - Window_Min)) * 255) #Normalize the intensities

    return New_Img

def mp(path_idx_save):
        
        path = path_idx_save[0]
        idx = path_idx_save[1]
        savePath = path_idx_save[2]
        
        img = Dicom_to_Image(path)
#         print(img.shape[0], img.shape[1])
        label = path.split("/")[-2]
#         cv2.imwrite(savePath+label+"/"+label+"_"+str(idx)+".png", img)

        return 0, 0

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
