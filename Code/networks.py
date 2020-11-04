"""
CAPSTONE PROJECT
TOPIC: An Interpretable Machine Learning Model for Gender Prediction With SHAP and CAM
AUTHOR: Renzo Castagnino
DATE: September 2020
"""

# %% ------------------------------------------- Imports ---------------------------------------------------------------
from pathlib import Path
import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F

# %% ------------------------------------------- Hyper Parameters ------------------------------------------------------
DROPOUT = 0.5


# %% ------------------------------------------- CNN Class -------------------------------------------------------------
class Linear_Model(nn.Module):
    def __init__(self):
        super(Linear_Model, self).__init__()
        # img = images
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        dim = x.shape[0]
        v = x.view(dim, 512, -1)
        x = v.mean(2)
        x = x.view(1, dim, 512)
        x = self.fc(x)
        return x.view(-1, 2)

# %% ------------------------------------------- Load Pre-Trained Model ------------------------------------------------
def load_pretrained_model(name='VGG16'):
    models_path = '../PreTrained_Models'
    Path(models_path).mkdir(parents=True, exist_ok=True)

    print('Loading Model...')

    if name == 'VGG16':
        try:
            model = torch.load(models_path + '/VGG16.pth')

        except:
            model = models.vgg16(pretrained=True, progress=True)
            torch.save(model, models_path + '/VGG16.pth')

        for i, param in enumerate(model.features.parameters()):
            if i == 24:
                break
            param.requires_grad = False

        model.classifier = torch.nn.Sequential(Linear_Model())

    elif name == 'Inception':
        try:
            model = torch.load(models_path + '/Inception.pth', aux_logits=False)

        except:
            model = models.inception_v3(pretrained=True, progress=True, aux_logits=False)

            torch.save(model, models_path + '/Inception.pth')
        model.fc.out_features = 2
        for i, param in enumerate(model.parameters()):
            if i == 200:
                break
            param.requires_grad = False

    elif name == 'ResNet':
        try:
            model = torch.load(models_path + '/ResNet.pth', aux_logits=False)

        except:
            model = models.resnet18(pretrained=True, progress=True)

            torch.save(model, models_path + '/ResNet.pth')
        model.fc.out_features = 2
        for i, param in enumerate(model.parameters()):
            if i == 45:
                break
            param.requires_grad = False

    print('Model Loaded' + name)
    return model
