import os
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def lineCat(mode, *args):
    lines = []
    bests = []

    for arg in args:
        lines.append(arg)
        if mode == "loss":
            bests.append([np.argmax(np.array(arg)), max(arg)])
        else mode == "acc":
            bests.append([np.argmax(np.array(arg)), max(arg)])

    return lines, bests

def draw(lines, bests, config, savePath, y_label, lineLabel):
    
    assert len(lines) == len(lineLabel), "Missmatch length!" 

    color = ["red", "blue", "yellow", "green", "purple", "black"]
    matplotlib.use("AGG") 
    plt.figure()

    # Draw lines
    for lineIdx in range(len(lines)):
        plt.plot(lines[lineIdx], c=color[lineIdx], label=lineLabel[lineIdx])

    plt.legend()

    # Draw best point
    for pointIdx in range(len(bests)):
        plt.scatter(bests[pointIdx][0], bests[pointIdx][1], c="black", marker='.')
        plt.text(bests[pointIdx][0], bests[pointIdx][1], "{}: {}; Epoch: {}".format(y_label, bests[pointIdx][1], bests[pointIdx][0]))

    # Words
    title = config["model"]+"_"+config["optimizer"]["name"]+str(config["optimizer"]["lr"])+"_batch"+str(config["batch_size"])+"_"+y_label
    if config["train_last"]:
        title += "_trainLast"
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(y_label)

    # Save plot
    plt.savefig(savePath+title+".png")

def saveInfo(config, acc, loss, epoch, savePath):
    if config["train_last"]:
        saveFile = savePath+config["model"]+"_"+config["optimizer"]["name"]+str(config["optimizer"]["lr"])+"_batch"+str(config["batch_size"])+"_trainLast"+".txt"
    else:
        saveFile = savePath+config["model"]+"_"+config["optimizer"]["name"]+str(config["optimizer"]["lr"])+"_batch"+str(config["batch_size"])+".txt"

    with open(saveFile, "w") as f:
        f.write("Best performance on validation set:\n")
        f.write("\tAccuracy: {}\n".format(str(acc)))
        f.write("\tLoss: {}\n".format(str(loss)))
        f.write("\tEpoch: {}\n".format(str(epoch)))

def writePred(names, preds, savePath):
    assert len(names) == len(preds), "Mismatch length!"

    with open(savePath+"result.csv", "w", newline='') as csvfile: 
        writer = csv.writer(csvfile)

        for name, pred in sorted(zip(names, preds), key = lambda x: x[0]):
            writer.writerow([name, pred])

def readBest(config, savePath):
    if config["train_last"]:
        saveFile = savePath+config["model"]+"_"+config["optimizer"]["name"]+str(config["optimizer"]["lr"])+"_batch"+str(config["batch_size"])+"_trainLast"+".txt"
    else:
        saveFile = savePath+config["model"]+"_"+config["optimizer"]["name"]+str(config["optimizer"]["lr"])+"_batch"+str(config["batch_size"])+".txt"
    
    if not os.path.exists(saveFile):
        return 0

    with open(saveFile, "r") as f:
        for line in f.readlines():
            if line.startswith("\tAccuracy"):
                return float(line.split()[1])

if __name__ == "__main__":
    lines = [[1, 2, 3, 4, 5], [1, 4, 9, 16, 25]]
    bests = [[1, 2], [3, 16]]
   
#     a, b = lineCat([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])
#     print(a, b)

    writePred(["a", "b", "c"], [1, 2, 3], "./")

