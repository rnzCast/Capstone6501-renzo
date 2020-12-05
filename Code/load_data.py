# %% ------------------------------------------- Imports ---------------------------------------------------------------

import pandas as pd
import cv2
import shutil
import os
import random
import re
import math

# %% ------------------------------------------- Load CSV --------------------------------------------------------------
# reads the CSV with the labels
df = pd.read_csv('../csv/celeba_list_attr.csv')

# %% ------------------------------------------- Split data ------------------------------------------------------------
for i in df.values:
    path = i[0]  # i[0] corresponds to the first column in the .csv referencing the image name
    g = i[21]  # i[21] corresponds to the column where the male attribute is.
    image = cv2.imread('../img_align_celeba/' + str(path))
    try:
        print('copying images...')
        if g == 1.0:
            shutil.copy('../img_align_celeba/' + str(path),
                        '../data/train/male')
        else:
            shutil.copy('../img_align_celeba/' + str(path),
                        '../data/train/female')
    except:
        pass
print('images copied!')

# %% ------------------------------------------- Downsampling ----------------------------------------------------------
# print('Downsampling images...!')

files_female = [f for f in os.listdir('../data/train/female') if re.match(r'[0-9]+.*\.jpg', f)]
files_male = [f for f in os.listdir('../data/train/male') if re.match(r'[0-9]+.*\.jpg', f)]
n_samples_female = len(files_female)
n_samples_male = len(files_male)

files = random.sample(files_female, (n_samples_female - n_samples_male))  # Pick n random files
for file in files:
    f = os.path.join('../data/train/female/', file)
    os.remove(f)
print('Downsampling done.')

# %% ------------------------------------------- Train/Test Split ------------------------------------------------------
print('Train/Test split...')
test_size = math.ceil(n_samples_female * 0.2)
files_female = [f for f in os.listdir('../data/train/female') if re.match(r'[0-9]+.*\.jpg', f)]
files_male = [f for f in os.listdir('../data/train/male') if re.match(r'[0-9]+.*\.jpg', f)]

files_female = random.sample(files_female, test_size)  # Pick n random files to copy to test folder
files_male = random.sample(files_male, test_size)  # Pick n random files to copy to test folder

# split train and test by randomly copying train file to test file
for files in files_female:
    shutil.move('../data/train/female/' + files, '../data/test/female')

for files in files_male:
    shutil.move('../data/train/male/' + files, '../data/test/male')

print('Train/Test split done.')

