import os
import sys

import torch
import yaml
import tqdm
import torch.nn as nn
import torch.optim as optim
from functools import partial
from joblib import cpu_count
from torch.utils.data import DataLoader

from model import get_model, mod_model
from dataset import brainDataset
# from network import Net

class Trainer:
    def __init__(self, config, train: DataLoader, val: DataLoader):
        self.config = config
        self.trainLoader = train
        self.valLoader = val
        self.best_score = 0

        self._init_params()

    def train(self):    
        for epoch in range(config["num_epochs"]):
            self._run_epoch(epoch)
            self._eval(epoch)
            self.scheduler.step

            torch.save(self.net.state_dict(), "last_{}.h5".format(self.config["model"])) 

    def _run_epoch(self, epoch):
        self.net.train()

        for param_group in self.optimizer.param_groups:
            lr = param_group["lr"]

        tq = tqdm.tqdm(self.trainLoader, total=len(self.trainLoader)//self.config["batch_size"])
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
            tq.set_postfix(loss="Accuracy="+str(100*correct/total)+"%; "+"Loss="+str(running_loss/i))

        # close tqdm
        tq.close()

    def _eval(self, epoch):
        self.net.eval()

        tq = tqdm.tqdm(self.trainLoader, total=len(self.trainLoader)//self.config["batch_size"])
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
        print("Accuracy="+str(100*correct/total)+"%; Loss="+str(running_loss/i))
        
        # if current best model
        if (100*correct/total) > self.best_score:
            self.best_score = (100*correct/total)
            torch.save(self.net.state_dict(), "best_{}.h5".format(self.config["model"]))

        # close tqdm
        tq.close()
        
    def _get_optimizer(self, optimizer, params):
        name = optimizer["name"]
        lr = optimizer["lr"]

        if name == "SGD":
            opt = optim.SGD(params, lr=lr)
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
        self.net = mod_model(config["train"]["class"], self.net)
        if self.config["train"]["use_finetune"]:
            self.net = torch.load(self.config["train"]["model_path"])    
        self.net.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer(self.config["optimizer"], filter(lambda p: p.requires_grad, self.net.parameters()))
        self.scheduler = self._get_scheduler()


if __name__ == "__main__":

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "9"    

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

    # Metrics


