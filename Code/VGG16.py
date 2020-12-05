"""
CAPSTONE PROJECT
TOPIC: An Interpretable Machine Learning Model for Gender Prediction With SHAP and CAM
AUTHOR: Renzo Castagnino
DATE: September 2020
"""

# %% ------------------------------------------- Imports ---------------------------------------------------------------
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import warnings
from torch.utils.data import DataLoader
from utills import *
from networks import *
from pathlib import Path
warnings.filterwarnings("ignore")

# %% ------------------------------------------- Hyper Parameters ------------------------------------------------------
model_path = 'models/'
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
NUM_WORKERS = 8
LR = 0.001
N_EPOCHS = 10

# %% ------------------------------------------- Data Dir --------------------------------------------------------------
DATA_DIR = (str(Path(__file__).parents[1]) + '/data')


# %% ------------------------------------------- Data Preparation ------------------------------------------------------
train_transforms = transforms.Compose([transforms.Resize(250),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(10),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(250),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


train_data = datasets.ImageFolder(DATA_DIR + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(DATA_DIR + '/test', transform=test_transforms)
print(len(train_data), len(test_data))


# %% ------------------------------------------- Data Loaders ----------------------------------------------------------
train_loader = DataLoader(train_data,
                          batch_size=TRAIN_BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)

test_loader = DataLoader(test_data,
                         batch_size=TEST_BATCH_SIZE,
                         shuffle=True,
                         num_workers=NUM_WORKERS)


# %% ------------------------------------------- VGG16 Model -----------------------------------------------------------
if __name__ == '__main__':
    model = load_pretrained_model('VGG16')  # channels of images
    criterion = nn.CrossEntropyLoss()  # defining loss function
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    vgg16_fc_only_exp_history, vgg16_fc_only_exp_best_model = train(model=model,
                                                                    criterion=criterion,
                                                                    optimizer=optimizer,
                                                                    train_loader=train_loader,
                                                                    validation_loader=test_loader,
                                                                    save_path='VGG16.pth',
                                                                    epochs=N_EPOCHS,
                                                                    lr_scheduler=scheduler)

