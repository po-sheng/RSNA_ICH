import yaml
import torch.nn as nn
import torchvision.models as models

def get_model(modelName):
    if modelName.startswith("resnet"):
        if modelName.endswith("18"):
            net = models.resnet18(pretrained=True, progress=True)
        elif modelName.endswith("34"):
            net = models.resnet34(pretrained=True, progress=True)
        elif modelName.endswith("50"):
            net = models.resnet50(pretrained=True, progress=True)
        elif modelName.endswith("101"):
            net = models.resnet101(pretrained=True, progress=True)
        elif modelName.endswith("152"):
            net = models.resnet152(pretrained=True, progress=True)
    
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

def mod_model(class2idx, train_last, modelName, model):
    # Change the input channel of first layer and
    # change the output feature number of last layer
    num_layers = len(list(model.named_modules()))
    num_classes = len(class2idx)
    count = 0
    first = 0
    
    for name, module in model.named_modules():
#         if "conv" in name and first == 0:
#             first = 1
#             outC = module.out_channels
#             kSize = module.kernel_size
#             stride = module.stride
#             pad = module.padding
#             dilat = module.dilation
#             gr = module.groups
#             bias = module.bias
#             
#             if modelName.startswith("resnet"):
#                 model.module.conv1 = nn.Conv2d(1, outC, kSize, stride, pad, dilat, gr, bias)
#             elif modelName.startswith("densenet"):
#                 model.module.features.conv0 = nn.Conv2d(1, outC, kSize, stride, pad, dilat, gr, bias)

        if count == num_layers - 1:
            inF = module.in_features
            if modelName.startswith("resnet"):
                model.fc = nn.Linear(in_features=inF, out_features=num_classes)
            elif modelName.startswith("densenet"):
                model.module.features.classifier = nn.Linear(in_features=inF, out_features=num_classes)
        count += 1

    # Set last block require grad
    if train_last:
        start = 0
        for name, param in model.named_parameters():
            if start == 1 or name.startswith("module.layer4"):
                param.requires_grad = True
                start = 1
            else:
                param.requires_grad = False

    return model

if __name__ == "__main__":
    net = nn.DataParallel(models.resnet34(pretrained=True, progress=True))

    mod_model({"a": 1, "b": 2}, net)

