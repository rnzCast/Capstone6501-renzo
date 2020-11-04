"""
CAPSTONE PROJECT
TOPIC: An Interpretable Machine Learning Model for Gender Prediction With SHAP and CAM
AUTHOR: Renzo Castagnino
DATE: September 2020
"""

# %% ------------------------------------------- Imports ---------------------------------------------------------------
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import warnings
from utills import *
from networks import *
warnings.filterwarnings("ignore")

# %% ------------------------------------------- Hyper Parameters ------------------------------------------------------
best_model_path = 'models/'
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
NUM_WORKERS = 8
LR = 0.001
N_EPOCHS = 10

# %% ------------------------------------------- Data Dir --------------------------------------------------------------
DATA_DIR = (str(Path(__file__).parents[1]) + '/data')

# %% ------------------------------------------- Data Preparation ------------------------------------------------------
train_transforms = transforms.Compose([transforms.Resize(300),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(10),
                                       transforms.CenterCrop(299),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(300),
                                      transforms.CenterCrop(299),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(DATA_DIR + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(DATA_DIR + '/test', transform=test_transforms)
print(len(train_data), len(test_data))

"""
size = len(train_data)
# split test_data into test and val data
train_data, test_data = torch.utils.data.random_split(train_data, [math.ceil(size * 0.8), int(size * 0.2)])

"""

# %% ------------------------------------------- Data Loaders ----------------------------------------------------------
train_loader = DataLoader(train_data,
                          batch_size=TRAIN_BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)

test_loader = DataLoader(test_data,
                         batch_size=TEST_BATCH_SIZE,
                         shuffle=True,
                         num_workers=NUM_WORKERS)


# validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=TEST_BATCH_SIZE, shuffle=True,
# num_workers=NUM_WORKERS)

# %% ------------------------------------------- Inception Model -------------------------------------------------------

if __name__ == '__main__':
    model = load_pretrained_model('Inception')  # channels of images
    criterion = nn.CrossEntropyLoss()  # defining loss function
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model_fc_only_exp_history, model_fc_only_exp_best_model = train(model=model,
                                                                    criterion=criterion,
                                                                    optimizer=optimizer,
                                                                    train_loader=train_loader,
                                                                    validation_loader=test_loader,
                                                                    save_path='Inception.pth',
                                                                    epochs=N_EPOCHS,
                                                                    lr_scheduler=scheduler)

