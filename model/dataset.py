import os
import glob
import torch
import pydicom
import numpy as np
from PIL import ImageStat as pilStat
import torchvision.transforms as trns
from torch.utils.data.dataset import Dataset

def getLabel(imgs, class2idx):
    label = []
    for img in imgs:
        fileName = os.path.basename(img)
        label.append(class2idx[fileName.split("_")[0]])
    
    return label

def readImg(path):
    
    # Show the attributes of ds 
    ds = pydicom.read_file(path)
    # print(ds.dir("pat"))

    # Binary file for raw image
    pixel_bytes = ds.PixelData
    # print(type(pixel_bytes))              # byte type

    # Get image matrix
    img = ds.pixel_array                    # ndarray

    return img

def readImg_refine(Path):
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

#     New_Img = np.zeros((rows, cols), np.uint8)
    Pixels = DCM_Img.pixel_array

    img = Pixels * Rescale_Slope + Rescale_Intercept
    img[img > Window_Max] = 255
    img[img > Window_Min] = 0
    img = 255 * (img - Window_Min) / (Window_Max - Window_Min)

#     for i in range(0, rows):
#         for j in range(0, cols):
#             Pix_Val = Pixels[i][j]
#             Rescale_Pix_Val = Pix_Val * Rescale_Slope + Rescale_Intercept
# 
#             if (Rescale_Pix_Val > Window_Max): #if intensity is greater than max window
#                 New_Img[i][j] = 255
#             elif (Rescale_Pix_Val < Window_Min): #if intensity is less than min window
#                 New_Img[i][j] = 0
#             else:
#                 New_Img[i][j] = (255 * (Rescale_Pix_Val - Window_Min) / (Window_Max - Window_Min)) #Normalize the intensities
    
    return img.astype(np.ubyte)

def preProcess(img):
    # Prevent overflow
#     img = img.astype(np.float)

    # Normalize to 0~255 and resize nparray to H*W*C  
#     maxNum = np.max(img)
#     minNum = np.min(img)

#     img = ((img - minNum) / (maxNum - minNum) * 255).astype(np.ubyte)
    img.reshape((img.shape[0], img.shape[1], 1))

    return img

def transform(img, phrase):
    if phrase == "train":
        transform = trns.Compose([
            trns.ToPILImage(),
#             trns.Resize((224, 224)),
            trns.Resize((512, 512)),
            trns.RandomHorizontalFlip()
        ])
    else:
        transform = trns.Compose([
            trns.ToPILImage(),
#             trns.Resize((224, 224))
            trns.Resize((512, 512))
        ])

    return transform(img)

def normalize(img):
    norm = trns.Compose([
        trns.ToTensor(),
        trns.Normalize(0.5, 0.5)
    ])

    return norm(img)

class brainDataset(Dataset):
    def __init__(self, dataset):
        self.dataPath = dataset["data"]
        self.phrase = dataset["phrase"]
        self.class2idx = dataset["class"]

        # list all the data file
        self.imgs = glob.glob(self.dataPath)
        if self.phrase != "test":
            self.lbls = getLabel(self.imgs, self.class2idx)
            assert len(self.imgs) == len(self.lbls), "mismatched length!"
        
        print("Total data in {} split: {}".format(self.phrase, len(self.imgs)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = readImg_refine(self.imgs[idx])               # Output ndarray
        
        if self.phrase == "test":
            lbl = self.imgs[idx]
        else:
            lbl = self.lbls[idx]
        
        img = preProcess(img)
        img = transform(img, self.phrase)           # Output PIL Image
        img = normalize(img)                        # Output Tensor
        
        return img, lbl

if __name__ == "__main__":
    torch.set_printoptions(threshold=np.inf)
    np.set_printoptions(threshold=np.inf)
    
    dataset = {"data": "../dataset/split/val/*.dcm", "phrase": "train", "class": {"epidural": 0, "healthy": 1, "intraparenchymal": 2, "intraventricular": 3, "subarachnoid": 4, "subdural": 5}}

    brain = brainDataset(dataset)

    print(np.array(brain[0][0]))
