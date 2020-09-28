"""
CAPSTONE PROJECT
TOPIC:
AN INTERPRETABLE MACHINE LEARNING MODEL FOR GENDER
PREDICTION USING SHAP AND CLASS ACTIVATION MAPS

AUTHOR: Renzo Castagnino
DATE: September 2020
"""

# %% ------------------------------------------- IMPORT PACKAGES -------------------------------------------------------
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path
import scipy.io


# %% ------------------------------------------- DATA DIR---------------------------------------------------------------
DATA_DIR = (str(Path(__file__).parents[1]) + '/data/')

mat = scipy.io.loadmat(str(Path(__file__).parents[1]) + '/data/imdb/imdb.mat')
print(type(mat))
print(mat.keys())
print(mat['imdb'])

for key in mat.keys():
    if item in mat[key]:
        print(key)


# %% ------------------------------------------- PREPROCESS ------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


data = ImageFolder(root=DATA_DIR, transform=preprocess)
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0)
# print(type(data))
# print(len(data))


# %% ------------------------------------------- MODEL VGG16------------------------------------------------------------
model_VGG16 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)


# %% ------------------------------------------- MOVE MODEL TO GPU------------------------------------------------------
# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model_VGG16.to('cuda')
