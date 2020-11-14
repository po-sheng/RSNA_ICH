import yaml
import torch.nn as nn
import torchvision.models as models

def get_model(modelName):
    if modelName.startswith("resnet"):
        if modelName.endswith("50"):
            net = models.resnet50(pretrained=True, progress=True)
        elif modelName.endswith("101"):
            net = models.resnet101(pretrained=True, progress=True)
        elif modelName.endswith("152"):
            net = models.resnet151(pretrained=True, progress=True)
    
    elif modelName.startswith("vgg"):
        if modelName.endswith("16"):
            net = models.vgg16(pretrained=True, progress=True)
        elif modelName.endswith("16_bn"):
            net = models.vgg16_bn(pretrained=True, progress=True)
        elif modelName.endswith("19"):
            net = models.vgg19(pretrained=True, progress=True)
        elif modelName.endswith("19_bn"):
            net = models.vgg19_bn(pretrained=True, progress=True)
    
    elif modelName.startswith("densenet"):
        if modelName.endswith("121"):
            net = models.densenet121(pretrained=True, progress=True)
        elif modelName.endswith("169"):
            net = models.densenet169(pretrained=True, progress=True)
        elif modelName.endswith("161"):
            net = models.densenet161(pretrained=True, progress=True)
 
    else:
        raise ValueError("Network [%s] not recognized." % modelName)

    return nn.DataParallel(net)

def mod_model(class2idx, model):
    # Change the input channel of first layer and
    # change the output feature number of last layer
    num_layers = len(list(model.named_modules()))
    num_classes = len(class2idx)
    count = 0
    first = 0
    
    for name, module in model.named_modules():
        if name.startswith("module.conv") and first == 0:
            first = 1
            outC = module.out_channels
            kSize = module.kernel_size
            stride = module.stride
            pad = module.padding
            dilat = module.dilation
            gr = module.groups
            bias = module.bias
            model.module.conv1 = nn.Conv2d(1, outC, kSize, stride, pad, dilat, gr, bias)
        
        if count == num_layers - 1:
            inF = module.in_features
            model.fc = nn.Linear(in_features=inF, out_features=num_classes)
        count += 1
    
    # Set last block require grad
#     for name, param in model.named_parameters():
#         print(name)

    return model

if __name__ == "__main__":
    net = nn.DataParallel(models.resnet50(pretrained=True, progress=True))

    mod_model({"a": 1, "b": 2}, net)

