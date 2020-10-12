"""
CAPSTONE PROJECT
TOPIC:
AN INTERPRETABLE MACHINE LEARNING MODEL FOR GENDER
PREDICTION USING SHAP AND CLASS ACTIVATION MAPS

AUTHOR: Renzo Castagnino
DATE: September 2020
"""

import copy
import os
import time
# %% ------------------------------------------- IMPORT PACKAGES -------------------------------------------------------
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torchvision import transforms


# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
MODEL_NAME = "inception"
N_CLASSES = 2
BATCH_SIZE = 8
N_EPOCHS = 2
INPUT_SIZE = 0
feature_extract = True

# %% ------------------------------------------- DATA DIR---------------------------------------------------------------
DATA_DIR = (str(Path(__file__).parents[1]) + '/data/data/')

# %% ------------------------------------------- PREPROCESS ------------------------------------------------------------
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'test']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True, num_workers=4) for x in ['train', 'test']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# %% ------------------------------------------- MODEL INCEPTION--------------------------------------------------------
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# %% -------------------------------------- Training Preparation--------------------------------------------------------
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# %% -------------------------------------- Training Loop --------------------------------------------------------------
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# %% -------------------------------------- Optimizers -----------------------------------------------------------------

model_ft, INPUT_SIZE = initialize_model(MODEL_NAME, N_CLASSES, feature_extract, use_pretrained=True)
print(model_ft)

criterion = nn.CrossEntropyLoss()

params_to_update = model_ft.parameters()
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=N_EPOCHS,
                             is_inception=(MODEL_NAME == "inception"))

model_ft = model_ft.to(device)
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# %% -------------------------------------- Plot Training --------------------------------------------------------------
ohist = []
ohist = [h.cpu().numpy() for h in hist]

plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1, N_EPOCHS + 1), ohist, label="Pretrained")
# plt.plot(range(1,num_epochs+1), shist,label="Scratch")
plt.ylim((0, 1.))
plt.xticks(np.arange(1, N_EPOCHS + 1, 1.0))
plt.legend()
plt.show()

# %% -------------------------------------- Save Model -----------------------------------------------------------------
checkpoint = {
    'parameters': model_ft.parameters,
    'state_dict': model_ft.state_dict()
}
torch.save(checkpoint, 'gender_inception.pth')

# %% -------------------------------------- Predict--- -----------------------------------------------------------------


# def image_transform(imagepath):
#     test_transforms = transforms.Compose([transforms.Resize(224),
#                                           transforms.CenterCrop(224),
#                                           transforms.ToTensor(),
#                                           transforms.Normalize([0.485, 0.456, 0.406],
#                                                                [0.229, 0.224, 0.225])])
#     image = Image.open(imagepath)
#     imagetensor = test_transforms(image)
#     return imagetensor
#
#
# def predict(imagepath, verbose=True):
#     if not verbose:
#         warnings.filterwarnings('ignore')
#
#     model_path = str(Path(__file__).parents[1]) + '/code/gender_inception.pth'
#     try:
#         checks_if_model_is_loaded = type(model_ft)
#     except:
#         model_ft.load_state_dict(torch.load(model_path), map_location='GPU')
#         model_ft.cuda()
#     model_ft.eval()
#     if verbose:
#         print("Model Loaded..")
#     image = image_transform(imagepath)
#     image1 = image[None, :, :, :].cuda()
#     ps = torch.exp(model_ft(image1))
#     topconf, topclass = ps.topk(1, dim=1)
#     if topclass.item() == 1:
#         return {'class': 'male', 'confidence': str(topconf.item())}
#     else:
#         return {'class': 'female', 'confidence': str(topconf.item())}
#
#
# print(predict(str(Path(__file__).parents[1]) + '/data/data/test/img01.png'))
