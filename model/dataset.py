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

def preProcess(img):
    # Prevent overflow
    img = img.astype(np.float)

    # Normalize to 0~255 and resize nparray to H*W*C  
    maxNum = np.max(img)
    minNum = np.min(img)

    img = ((img - minNum) / (maxNum - minNum) * 255).astype(np.ubyte)
    img.reshape((img.shape[0], img.shape[1], 1))

    return img

def transform(img, phrase):
    if phrase == "train":
        transform = trns.Compose([
            trns.ToPILImage(),
            trns.Resize((224, 224)),
            trns.RandomHorizontalFlip()
        ])
    else:
        transform = trns.Compose([
            trns.ToPILImage(),
            trns.Resize((224, 224)),
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
        self.lbls = getLabel(self.imgs, self.class2idx)
        assert len(self.imgs) == len(self.lbls), "mismatched length!"
        
        print("Total data in {} split: {}".format(self.phrase, len(self.imgs)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = readImg(self.imgs[idx])               # Output ndarray
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

    print(np.array(brain[1][0]))
