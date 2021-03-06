from torchvision import transforms, datasets
import shap
from utills import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.nn import Identity
import torch
is_gpu = torch.cuda.is_available()
DATA_DIR = (str(Path(__file__).parents[1]) + '/data/')
device = torch.device("cuda:0" if is_gpu else "cpu")

# select the folder of the images to be analyze
folder_name = 'val_male'

# select the image to be analyze by SHAP
img_name = '2.jpg'


# Select the model to run: 1 for VGG16, 2 for Inception, 3 for ResNet
model_to_run = 1

print('Loading Model...')
model = None
test_transforms = None
layer = None
shape = None

inp_user = model_to_run
if inp_user == 1:
    model = load_model('../models/VGG16.pth')
    layer = model.features[7]
    shape = (224, 224)
    save_img_name = 'SHAP_VGG16_' + folder_name + '_' + img_name
    test_transforms = transforms.Compose([transforms.Resize(250),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
elif inp_user == 2:
    model = load_model('../models/Inception.pth')
    model.dropout = Identity()
    layer = model.Conv2d_4a_3x3.conv
    shape = (299, 299)
    save_img_name = 'SHAP_Inception_- ' + folder_name + '_' + img_name
    test_transforms = transforms.Compose([transforms.Resize(300),
                                          transforms.CenterCrop(299),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
elif inp_user == 3:
    model = load_model('../models/ResNet.pth')
    layer = model.layer2[0].conv2
    shape = (299, 299)
    save_img_name = 'SHAP_ResNet_' + folder_name + '_' + img_name
    test_transforms = transforms.Compose([transforms.Resize(300),
                                          transforms.CenterCrop(299),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
print('Model Loaded')


model.eval()
model.to(device)

print("Reading images")
test_data = datasets.ImageFolder(DATA_DIR + '/test', transform=test_transforms)
print("Reading images Completed")

test_loader = torch.utils.data.DataLoader(test_data, batch_size=100)
test_images = next(iter(test_loader))[0]  # .numpy()  # getting only first batch
print(len(test_images), "Batch of images is selected")


X_image = Image.open('../' + folder_name + '/' + img_name)
X = test_transforms(X_image)
X = X.unsqueeze(0)

# image for printing
img = cv2.imread('../' + folder_name + '/' + img_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)
img = cv2.resize(img, shape, 2)
img = img / 255
img = np.expand_dims(img, axis=0)
print(img.shape)

print("Training Gradient Explainer...")
e_explainer = shap.GradientExplainer((model, layer), test_images.to(device))
print("Calculating Shap Values of given Images...")
shap_values, indexes = e_explainer.shap_values(X.to(device), ranked_outputs=2, nsamples=200)
shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

# get the names for the classes
# our classes
classes = {0: 'Female', 1: 'Male'}
index_names = np.vectorize(lambda i: classes[i])(indexes.cpu())

shap.image_plot(shap_values, img, index_names, show=False)
plt.savefig('../image_results/' + save_img_name)
print('The image ', save_img_name, 'has been Saved!')
