"""
CAPSTONE PROJECT
TOPIC:
AN INTERPRETABLE MACHINE LEARNING MODEL FOR GENDER
PREDICTION USING SHAP AND CLASS ACTIVATION MAPS

AUTHOR: Renzo Castagnino
DATE: September 2020
"""


# %% -------------------------------------------Imports ----------------------------------------------------------------
import time
from pathlib import Path
from PIL import ImageFile
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from matplotlib import pyplot as plt
import torch
import warnings
warnings.filterwarnings("ignore")


# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 0.001
BATCH_SIZE = 4
DROPOUT = 0.1
N_EPOCHS = 10
STEPS = 0
RUNNING_LOSS = 0


# %% ------------------------------------------- Data Dir --------------------------------------------------------------
DATA_DIR = (str(Path(__file__).parents[1]) + '/data/data/')


# %% ------------------------------------------- Data Preparation-------------------------------------------------------
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


train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)


# %% ------------------------------------------- VGG16 Model -----------------------------------------------------------
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
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
    model.to(device)

    for ii, (inputs, labels) in enumerate(train_loader):
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

# Freeze parameters so we don't do backpropagation
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

training_losses = []
testing_losses = []
test_accuracy = []
total_steps = []
print_every = 1
for epoch in range(N_EPOCHS):
    print('training...')
    for inputs, labels in train_loader:
        STEPS += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        RUNNING_LOSS += loss.item()

        if STEPS % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            training_losses.append(RUNNING_LOSS / print_every)
            testing_losses.append(test_loss / len(test_loader))
            test_accuracy.append(accuracy / len(test_loader))
            total_steps.append(STEPS)
            print(f"Device {device}.."
                  f"Epoch {epoch + 1}/{N_EPOCHS}.. "
                  f"Step {STEPS}.. "
                  f"Train loss: {RUNNING_LOSS / print_every:.3f}.. "
                  f"Test loss: {test_loss / len(test_loader):.3f}.. "
                  f"Test accuracy: {accuracy / len(test_loader):.3f}")
            RUNNING_LOSS = 0
            model.train()


# %% -------------------------------------- Plot Training --------------------------------------------------------------
plt.plot(total_steps, training_losses, label='Train Loss')
plt.plot(total_steps, testing_losses, label='Test Loss')
plt.plot(total_steps, test_accuracy, label='Test Accuracy')
plt.legend()
plt.grid()
plt.show()

# %% -------------------------------------- Save Model -----------------------------------------------------------------
checkpoint = {
    'parameters': model.parameters,
    'state_dict': model.state_dict()
}

torch.save(checkpoint, 'gender_vgg.pth')


