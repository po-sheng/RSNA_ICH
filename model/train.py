import os
import sys

import torch
import yaml
import tqdm
import torch.nn as nn
from statistics import mean
import torch.optim as optim
from functools import partial
from joblib import cpu_count
from torch.utils.data import DataLoader

from metric import lineCat, draw, saveInfo, readBest
from model import get_model, mod_model 
from dataset import brainDataset

class Trainer:
    def __init__(self, config, train: DataLoader, val: DataLoader):
        self.config = config
        self.trainLoader = train
        self.valLoader = val
        self.best_score = readBest(self.config, "result/")
        self.trainL = []
        self.trainA = []
        self.valL = []
        self.valA = []

        self._init_params()

    def train(self):   
        for epoch in range(config["num_epochs"]):
            # Training
            self._run_epoch(epoch)

            # Validation
            self._eval(epoch)
            
            # Scheduler
            self.scheduler.step()

            torch.save(self.net.state_dict(), "checkpoint/last_{}.h5".format(self.config["model"]))  
            
            if len(self.trainL) > 10 and mean(self.trainL[-10:]) - self.trainL[-1] < 0.001:
                break;

        # Metric visualization
        # Loss
        lines, bests = lineCat(self.trainL, self.valL)
        draw(lines, bests, self.config, "result/", "Loss", ["train", "val"])

        # Accuracy
        lines, bests = lineCat(self.trainA, self.valA)
        draw(lines, bests, self.config, "result/", "Accuracy", ["train", "val"])
    
    def _run_epoch(self, epoch):
        self.net.train()

        for param_group in self.optimizer.param_groups:
            lr = param_group["lr"]

        tq = tqdm.tqdm(self.trainLoader, total=len(list(self.trainLoader)))
        tq.set_description("Epoch: {}, lr: {}".format(epoch, lr))
        
        i = 0
        total = 0
        correct = 0
        running_loss = 0
        for data in tq:
            # get the input; data is a list of [inputs, labels]
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda()

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # calculate statistics
            i += 1
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # print statistics
            tq.set_postfix(loss="{:.4f}".format(running_loss/i)+"; Accuracy="+"{:.4f}".format(100*correct/total)+"%")

        # close tqdm
        tq.close()

        # Metric
        self.trainL.append(running_loss/i)
        self.trainA.append(100*correct/total)

    def _eval(self, epoch):
        self.net.eval()

        tq = tqdm.tqdm(self.valLoader, total=len(list(self.valLoader)))
        tq.set_description("Validation")
        
        i = 0
        total = 0
        correct = 0
        running_loss = 0
        for data in tq:
            # get the input; data is a list of [inputs, labels]
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda()

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

            # calculate statistics
            i += 1
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # print statistics
        print("Loss="+"{:.4f}".format(running_loss/i)+"; Accuracy="+"{:.4f}".format(100*correct/total)+"%")
        
        # if current best model
        if (100*correct/total) > self.best_score:
            self.best_score = (100*correct/total)
            saveInfo(self.config, self.best_score, running_loss/i, epoch, "result/")
            torch.save(self.net.state_dict(), "checkpoint/best_{}.h5".format(self.config["model"]))

        # close tqdm
        tq.close()
        
        # Metric
        self.valL.append(running_loss/i)
        self.valA.append(100*correct/total)

        return running_loss/i, 100*correct/total
        
    def _get_optimizer(self, optimizer, params):
        name = optimizer["name"]
        lr = optimizer["lr"]

        if name == "SGD":
            opt = optim.SGD(params, momentum=0.9, lr=lr)
        elif name == "Adam":
            opt = optim.Adam(params, lr=lr)
        elif name == "Adadelta":
            opt = optim.Adadelta(params, lr=lr)
        else:
            raise ValueError("Optimizer [%s] not recognized." % name)
           
        return opt

    def _get_scheduler(self):
        sched = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config["scheduler"]["step_size"], gamma=self.config["scheduler"]["gamma"])

        return sched

    def _init_params(self):
        self.net = get_model(self.config["model"])
        self.net = mod_model(self.config["train"]["class"], self.config["train_last"], self.config["model"], self.net)
        if self.config["train"]["use_finetune"]:
            self.net = torch.load(self.config["train"]["model_path"])    
        self.net.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer(self.config["optimizer"], filter(lambda p: p.requires_grad, self.net.parameters()))
        self.scheduler = self._get_scheduler()


if __name__ == "__main__":

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6"    

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.cuda.empty_cache()

    # Read config
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    # Read dataset
    batch_size = config["batch_size"]
    get_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=cpu_count(), shuffle=True, drop_last=True)

    # Load data
    datasets = map(config.get, ("train", "val"))
    datasets = map(brainDataset, datasets)
    train, val = map(get_dataloader, datasets)

    # Train
    trainer = Trainer(config, train=train, val=val)
    trainer.train()

