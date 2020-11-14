import numpy as np
import matplotlib.pyplot as plt

def lineCat(*args):
    lines = []
    bests = []

    for arg in args:
        lines.append(arg)
        bests.append([np.argmax(np.array(arg)), max(arg)])

    return lines, bests

def draw(lines, bests, config, savePath, y_label, lineLabel):
    
    assert len(lines) == len(lineLabel), "Missmatch length!" 

    color = ["red", "blue", "yellow", "green", "purple", "black"]
    plt.figure()

    # Draw lines
    for lineIdx in range(len(lines)):
        plt.plot(lines[lineIdx], c=color[lineIdx], label=lineLabel[lineNum])

    plt.legend()

    # Draw best point
    for pointIdx in range(len(bests)):
        plt.scatter(bests[pointIdx][0], bests[pointIdx][1], c="black", marker='.')
        plt.text(bests[pointIdx][0], bests[pointIdx][1], "{}: {}; Epoch: {}".format(y_label, bests[pointIdx][1], bests[pointIdx][0]))

    # Words
    plt.title(config["model"]+"_"+config["optimizer"]["name"]+str(config["optimizer"]["lr"])+"_batch"+str(config["batch_size"]))
    plt.xlabel("Epochs")
    plt.ylabel(y_label)

    # Save plot
    plt.savefig(savePath+config["model"]+"_"+config["optimizer"]["name"]+str(config["optimizer"]["lr"])+"_batch"+str(config["batch_size"])+"_"+y_label+".png")

def saveInfo(config, acc, loss, epoch, savePath):
    with open(savePath+config["model"]+"_"+config["optimizer"]["name"]+str(config["optimizer"]["lr"])+"_batch"+str(config["batch_size"])+".txt", "w") as f:
        f.write("Best performance on validation set:\n")
        f.write("\tAccuracy: {}\n".format(str(acc)))
        f.write("\tLoss: {}\n".format(str(loss)))
        f.write("\tEpoch: {}\n".format(str(epoch)))

if __name__ == "__main__":
    lines = [[1, 2, 3, 4, 5], [1, 4, 9, 16, 25]]
    bests = [[1, 2], [3, 16]]
   
    a, b = lineCat([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])
    print(a, b)

