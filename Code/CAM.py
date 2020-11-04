from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import cv2
from utills import *
from torch.nn import Identity
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from torch.cuda import is_available
from torch import device
is_gpu = is_available()
device = device("cuda:0" if is_gpu else "cpu")

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

folder_path = '../val_images/'


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


def makeCAM(model_nm, image_used):

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())  # getting output shape of last conv layer

    print('Loading Model...')
    inp_user = model_nm
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

    img_pil = Image.fromarray(image_used.astype('uint8'), 'RGB')

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

    img = image_used
    height, width, _ = img.shape
    print("Creating CAM Image")
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    return result


def function_drive():
    all_images = list(os.walk(folder_path))[0][2]

    # Considering the size of each image as 229,
    big_image = np.array([299, 299])

    # Multiplying with the number of images at the end
    big_image[0] = np.multiply(big_image[0], all_images.__len__()) + 300
    big_image[1] = np.multiply(big_image[1], 4)
    # adding padding at each side of the image for 5 pixels
    big_image = np.dot(np.array([5, 5]), all_images.__len__()) + big_image
    big_image = np.append(big_image, 3)
    big_image = np.zeros(shape=big_image)
    j = 0

    for i in all_images:
        for k in range(4):
            image_to_send = plt.imread(folder_path+i)
            if k < 1:
                imag = cv2.resize(plt.imread(folder_path + i), (299, 299))
            else:
                imag = cv2.resize(makeCAM(model_nm = int(k), image_used = image_to_send),
                                  (299, 299))
            start_x = (j // 4) * (299 + 5) + 300
            start_y = (j % 4) * (299 + 5)

            big_image[start_x:start_x + 299, start_y:start_y + 299, :] = imag

            j += 1
    imag = cv2.resize(plt.imread(folder_path + all_images[0]), (299, 299), interpolation=3)
    big_image = np.array(big_image).astype('uint8')
    big_image = cv2.putText(big_image, "Image", (0, 290), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (255, 255, 255), 6)
    big_image = cv2.putText(big_image, "VGG", (299 + 10+40, 290), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (255, 255, 255), 6)
    big_image = cv2.putText(big_image, "Inception", ((299 * 2) + 10+30, 290), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (255, 255, 255), 6)
    big_image = cv2.putText(big_image, "ResNet", ((299 * 3) + 10 + 40, 290), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (255, 255, 255), 6)
    cv2.imwrite('../image_results/CAM4.jpeg', cv2.cvtColor(big_image, cv2.COLOR_BGR2RGB))


function_drive()
