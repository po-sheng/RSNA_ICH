import os
import yaml

import torch
import tqdm
import copy
from joblib import cpu_count
from torch.utils.data import DataLoader
from functools import partial

from model import get_model, mod_model
from metric import writePred 
from dataset import brainDataset

class Predictor:
    def __init__(self, config, test):
        self.config = config
        self.testLoader = test
        self.names = []
        self.preds = []

        self._init_params()

    def predict(self):
        self.net.eval()
        
        tq = tqdm.tqdm(self.testLoader, total=len(list(self.testLoader)))
        
        with torch.no_grad():
            for data in tq:
                inputs, paths = data
                inputs = inputs.cuda()

                outputs = self.net(inputs)
                _, predicted = torch.max(outputs, 1)

                self.preds += list(predicted.cpu().numpy())        
                for path in paths:
                    fileName = os.path.basename(path)
                    self.names.append(fileName.split(".")[0])

        return self.names, self.preds

    def _init_params(self):
        self.net = get_model(config["model"])
        self.net = mod_model(config["test"]["class"], self.net)
        self.net.load_state_dict(torch.load(config["test"]["model_path"]))
        self.net.cuda()

if __name__ == "__main__":
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"    

    # Read config
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    # Read dataset
    get_dataloader = partial(DataLoader, batch_size=1, num_workers=cpu_count(), shuffle=False)

    # Load data
    datasets = config.get("test")
    datasets = brainDataset(datasets)
    test = get_dataloader(datasets)

    # Train
    predictor = Predictor(config, test=test)
    names, preds = predictor.predict()

    # Convert predict idx to label
    pred_labels = []
    class2idx = config["test"]["class"]
    for pred in preds:
        for key, value in class2idx.items():
            if pred == value:
                pred_labels.append(key)

    writePred(names, pred_labels, "./")
