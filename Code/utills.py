"""
CAPSTONE PROJECT
TOPIC: An Interpretable Machine Learning Model for Gender Prediction With SHAP and CAM
AUTHOR: Renzo Castagnino
DATE: September 2020
"""

# %% ------------------------------------------- Imports ---------------------------------------------------------------
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import copy
from pathlib import Path
import numpy as np
is_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if is_gpu else "cpu")

# %% ------------------------------------------- Prediction ------------------------------------------------------------
def predict(model, data_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for data in data_loader:
            images, classes = data
            images, classes = images.to(device), classes.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.append(predicted)
            y_true.append(classes)
    y_pred = torch.cat(y_pred).cpu()
    y_true = torch.cat(y_true).cpu()
    if is_gpu:
        torch.cuda.empty_cache()
    return y_pred, y_true


def accuracy(y_pred, y_true):
    return (y_pred == y_true).sum().item() / len(y_true)


# %% ------------------------------------------- Plots -----------------------------------------------------------------
def plot_loss_and_accuracy_curves(history):
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    ax[0].plot(history['train_accuracy'], color='royalblue', label='Train Accuracy')
    ax[0].plot(history['test_accuracy'], color='darkred', label='Test Accuracy')

    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_title('Training and Test Accuracy')
    ax[0].legend()

    ax[1].plot(history['train_loss'], color='royalblue', label='Train Loss')
    ax[1].plot(history['test_loss'], color='darkred', label='Test Loss')

    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Cross Entropy Loss')
    ax[1].set_title('Training and Test Losses')
    ax[1].legend()
    plt.show()
    plt.savefig('../plots/accuracy_loss4.png')

# %% ------------------------------------------- Save Model ------------------------------------------------------------
def save_model(model_state, model_path):
    torch.save(model_state, model_path)

# %% ------------------------------------------- Training Model --------------------------------------------------------
def train(model, criterion, optimizer, train_loader, validation_loader, save_path=None, epochs=10, plot_curves=True,
          history=None, lr_scheduler=None):

    Path("../models").mkdir(parents=True, exist_ok=True)
    model.to(device)
    best_model = copy.deepcopy(model)

    if history is None:
        history = {'train_loss': [], 'test_loss': [], 'train_accuracy': [], 'test_accuracy': [],
                   'epochs': 0, 'best_test_accuracy': -np.inf, 'min_test_loss': np.inf}

    prev_epochs = history['epochs']
    best_test_accuracy = history['best_test_accuracy']
    min_test_loss = history['min_test_loss']

    total_epochs = prev_epochs + epochs

    for epoch in range(prev_epochs, total_epochs):  # loop over the dataset multiple times

        train_loss = 0.0
        test_loss = 0.0

        train_accuracy = 0.0
        test_accuracy = 0.0

        correct = 0
        model.train()
        lr_scheduler.step()

        outer = tqdm(total=len(train_loader.dataset), desc='Train Epoch: %s / %s' % (epoch + 1, total_epochs),
                     position=0, leave=True)

        for inputs, classes in train_loader:
            inputs, classes = inputs.to(device), classes.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)  # ----> forward pass
            loss = criterion(outputs, classes)  # ----> compute loss

            loss.backward()  # ----> backward pass
            optimizer.step()  # ----> weights update

            # print statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == classes).sum().item()
            outer.update(len(inputs))
        outer.close()
        train_accuracy = correct / len(train_loader.dataset)
        train_loss = train_loss / len(train_loader)

        # Test loss and accuracy
        with torch.no_grad():
            correct = 0
            model.eval()

            for inputs, classes in validation_loader:
                inputs, classes = inputs.to(device), classes.to(device)
                outputs = model(inputs)  # ----> forward pass

                loss = criterion(outputs, classes)  # ----> compute loss
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == classes).sum().item()

        test_accuracy = correct / len(validation_loader.dataset)
        test_loss = test_loss / len(validation_loader)

        epoch_log = tqdm(total=0, bar_format='{desc}', position=0, leave=True)
        epoch_log.set_description_str(
            'LR: {} | Train Loss: {:.6f} | Validation Loss: {:.6f} | Train Accuracy: {:.2f} | Validation Accuracy: {:.2f}'.format(
                lr_scheduler.get_lr(), train_loss, test_loss, train_accuracy, test_accuracy))
        epoch_log.close()

        # saving best model params for minimum validation loss during all epochs
        if test_accuracy >= best_test_accuracy:
            best_test_accuracy = test_accuracy
            min_test_loss = test_loss
            best_model.load_state_dict(model.state_dict())

            torch.save(best_model, '../models/' + save_path)

        history['epochs'] += 1
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_accuracy'].append(train_accuracy)
        history['test_accuracy'].append(test_accuracy)
        history['min_test_loss'] = min_test_loss
        history['best_test_accuracy'] = best_test_accuracy

    if plot_curves:
        plot_loss_and_accuracy_curves(history)

    if is_gpu:
        torch.cuda.empty_cache()
    return history, best_model

# %% ------------------------------------------- Load Model ------------------------------------------------------------
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))  # used to run it locally, no GPU.
    # model = torch.load(model_path)
    return model

