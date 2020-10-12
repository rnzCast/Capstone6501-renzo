"""
CAPSTONE PROJECT
TOPIC:
AN INTERPRETABLE MACHINE LEARNING MODEL FOR GENDER
PREDICTION USING SHAP AND CLASS ACTIVATION MAPS

AUTHOR: Renzo Castagnino
DATE: September 2020
"""

import time
# %% ------------------------------------------- IMPORT PACKAGES -------------------------------------------------------
from pathlib import Path
import torch
from PIL import ImageFile
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import warnings
warnings.filterwarnings("ignore")

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 0.001
N_EPOCHS = 500
BATCH_SIZE = 4
DROPOUT = 0.1

# %% ------------------------------------------- DATA DIR---------------------------------------------------------------
DATA_DIR = (str(Path(__file__).parents[1]) + '/data/data/')


# %% ------------------------------------------- PREPROCESS ------------------------------------------------------------
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(DATA_DIR + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(DATA_DIR + '/test', transform=test_transforms)


trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

# %% ------------------------------------------- MODEL VGG16------------------------------------------------------------
model = models.vgg16(pretrained=True)
print(model)

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2))
# model.classifier = classifier

# %% -------------------------------------- Training Preparation--------------------------------------------------------
for device in ['cpu']:

    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    model.to(device)

    for ii, (inputs, labels) in enumerate(trainloader):
        print(ii)
        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        start = time.time()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if ii == 3:
            break

    print(f"Device = {device}; Time per batch: {(time.time() - start) / 3:.3f} seconds")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.2),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2))

criterion = torch.nn.CrossEntropyLoss()
cost = torch.nn.CrossEntropyLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
model.to(device)

# %% -------------------------------------- Training Loop --------------------------------------------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True

traininglosses = []
testinglosses = []
testaccuracy = []
totalsteps = []
epochs = 10
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    print('training...')
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            traininglosses.append(running_loss / print_every)
            testinglosses.append(test_loss / len(testloader))
            testaccuracy.append(accuracy / len(testloader))
            totalsteps.append(steps)
            print(f"Device {device}.."
                  f"Epoch {epoch + 1}/{epochs}.. "
                  f"Step {steps}.. "
                  f"Train loss: {running_loss / print_every:.3f}.. "
                  f"Test loss: {test_loss / len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy / len(testloader):.3f}")
            running_loss = 0
            model.train()


# %% -------------------------------------- Plot Training --------------------------------------------------------------
from matplotlib import pyplot as plt
plt.plot(totalsteps, traininglosses, label='Train Loss')
plt.plot(totalsteps, testinglosses, label='Test Loss')
plt.plot(totalsteps, testaccuracy, label='Test Accuracy')
plt.legend()
plt.grid()
plt.show()

# %% -------------------------------------- Save Model -- --------------------------------------------------------------
checkpoint = {
    'parameters': model.parameters,
    'state_dict': model.state_dict()
}

torch.save(checkpoint, 'gender_vgg.pth')
