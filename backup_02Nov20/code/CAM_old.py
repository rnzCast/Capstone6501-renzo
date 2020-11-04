"""
CAPSTONE PROJECT
TOPIC: An Interpretable Machine Learning Model for Gender Prediction With SHAP and CAM
AUTHOR: Renzo Castagnino
DATE: September 2020
"""

# %% ------------------------------------------- Imports ---------------------------------------------------------------
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import cv2
from utills import *
import numpy as np
from torch.nn import Identity
import torch
import matplotlib.pyplot as plt
is_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if is_gpu else "cpu")

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())  # getting output shape of last conv layer


def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        beforeDot = feature_conv.reshape((nc, h * w))
        cam = np.matmul(weight_softmax[idx], beforeDot)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


print('Loading Model...')
while True:
    inp_user = input('select the model\n\t 1. VGG16\n\t 2. Inception\n\t 3. ResNet\n'
                     '\nPlease Input: \n')
    try:
        inp_user = int(inp_user)
    except:
        continue
    if 0< inp_user < 4:
        break
    else:
        print(' You have Selected Wrong Option. Select Again.\n\n')

if inp_user == 1:
    model = load_model('../models/VGG16.pth')
    final_conv_name = 'features'
    test_transforms = transforms.Compose([transforms.Resize(250),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
elif inp_user == 2:
    model = load_model('../models/Inception.pth')
    model.dropout = Identity()
    final_conv_name = 'Mixed_7c'
    test_transforms = transforms.Compose([transforms.Resize(300),
                                          transforms.CenterCrop(299),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

elif inp_user == 3:
    model = load_model('../models/ResNet.pth')

    final_conv_name = 'layer4'
    test_transforms = transforms.Compose([transforms.Resize(300),
                                          transforms.CenterCrop(299),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
print('Model Loaded')

model.eval()
model.to(device)
# hook the feature extractor
features_blobs = []
model._modules.get(final_conv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].data.cpu().numpy())


img_path = '../../data/test/male/161034.jpg'
img_pil = Image.open(img_path)

img = test_transforms(img_pil)
img_variable = Variable(img.unsqueeze(0))
output = model(img_variable.to(device))

# our classes
classes = {0: 'female', 1: 'male'}

h_x = F.softmax(output, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.cpu().numpy()
idx = idx.cpu().numpy()
# output the prediction
for i in range(0, 2):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

# render the CAM and output
img = cv2.imread(img_path)
height, width, _ = img.shape
print("Creating CAM Image")
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('../CAM.jpg', result)
plt.show()

