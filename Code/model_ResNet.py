"""
CAPSTONE PROJECT
TOPIC:
AN INTERPRETABLE MACHINE LEARNING MODEL FOR GENDER
PREDICTION USING SHAP AND CLASS ACTIVATION MAPS

AUTHOR: Renzo Castagnino
DATE: September 2020
"""

# %% ------------------------------------ Imports ----------------------------------------------------------------------
from pathlib import Path
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import ImageFile
from matplotlib import pyplot as plt
import torch
import warnings
from PIL import Image

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 0.001
BATCH_SIZE = 4
DROPOUT = 0.1
N_EPOCHS = 10
STEPS = 0
RUNNING_LOSS = 0

# %% ------------------------------------------- Data Dir --------------------------------------------------------------
DATA_DIR = (str(Path(__file__).parents[1]) + '/data/data/')

# %% ------------------------------------------ Data Preparation -------------------------------------------------------
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
test_data = datasets.ImageFolder(DATA_DIR + '/val', transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

# %% ------------------------------------------- ResNet Model ----------------------------------------------------------
model = models.resnet50(pretrained=True)
print(model)

# Time to Freeze params of the downloaded model.
for params in model.parameters():
    params.requires_grad = False

classifier = nn.Sequential(nn.Linear(2048, 512),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(512, 2),
                           nn.LogSoftmax(dim=1))

model.fc = classifier

criterion = nn.NLLLoss()
optimiser = optim.Adam(model.fc.parameters(), lr=0.003)
device = 'cuda'
model.to(device)

# %% -------------------------------------- Training Preparation--------------------------------------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True
train_losses = []
test_losses = []
test_accuracy = []
total_steps = []

epochs = 2
steps = 0
runn = 0
print_every = 10

for epoch in range(epochs):
    for images, labels in train_loader:
        steps += 1

        images, labels = images.to(device), labels.to(device)

        optimiser.zero_grad()

        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()

        optimiser.step()

        runn += loss.item()

        if steps % print_every == 0:
            model.eval()
            test_loss = 0
            acc = 0

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                logps = model(images)
                loss = criterion(logps, labels)
                test_loss += loss.item()

                ps = torch.exp(logps)
                top_ps, top_c = ps.topk(1, dim=1)
                equal = top_c == labels.view(top_c.shape)
                acc += torch.mean(equal.type(torch.FloatTensor)).item()

            train_losses.append(runn / print_every)
            test_losses.append(test_loss / len(test_loader))
            test_accuracy.append(acc / len(test_loader))
            total_steps.append(steps)

            print(f"Epoch {epoch + 1}/{epochs}.. "
                  f"Train loss: {runn / print_every:.3f}.. "
                  f"Test loss: {test_loss / len(test_loader):.3f}.. "
                  f"Test accuracy: {acc / len(test_loader):.3f}")
            runn = 0
            model.train()

# %% -------------------------------------- Plot Training --------------------------------------------------------------
plt.plot(total_steps, train_losses, label='Train Loss')
plt.plot(total_steps, test_losses, label='Test Loss')
plt.plot(total_steps, test_accuracy, label='Test Accuracy')
plt.legend()
plt.grid()
plt.show()

# %% -------------------------------------- Save Model -----------------------------------------------------------------
checkpoint = {
    'parameters': model.parameters,
    'state_dict': model.state_dict()
}

torch.save(checkpoint, 'gender_ResNet.pth')


# %% -------------------------------------- Predict --------------------------------------------------------------------
# def image_transform(image_path):
#     test_transforms = transforms.Compose([transforms.Resize(224),
#                                           transforms.CenterCrop(224),
#                                           transforms.ToTensor(),
#                                           transforms.Normalize([0.485, 0.456, 0.406],
#                                                                [0.229, 0.224, 0.225])])
#     image = Image.open(image_path)
#     image_tensor = test_transforms(image)
#     return image_tensor
#
#
# def predict(image_path, verbose=True):
#     if not verbose:
#         warnings.filterwarnings('ignore')
#
#     model_path = str(Path(__file__).parents[1]) + '/code/gender_resnet.pth'
#     try:
#         checks_if_model_is_loaded = type(model)
#     except:
#         model.load_state_dict(torch.load(model_path), map_location='GPU')
#         model.cuda()
#     model.eval()
#     if verbose:
#         print("Model Loaded..")
#     image = image_transform(image_path)
#     image1 = image[None, :, :, :].cuda()
#     ps = torch.exp(model(image1))
#     topconf, topclass = ps.topk(1, dim=1)
#     if topclass.item() == 1:
#         return {'class': 'male', 'confidence': str(topconf.item())}
#     else:
#         return {'class': 'female', 'confidence': str(topconf.item())}
#
#
# print(predict(str(Path(__file__).parents[1]) + '/data/data/test/1.jpg'))
