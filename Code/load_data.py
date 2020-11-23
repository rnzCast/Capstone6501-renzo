import pandas as pd
import cv2
import shutil
from pathlib import Path


df = pd.read_csv('../csv/celeba_list_attr.csv')
print(df)

for i in df.values:
    path = i[0]  # i[0] corresponds to the first column in the .csv file, referencing the image name
    g = i[32]  # i[21] corresponds to the column where the male attribute is.
    image = cv2.imread('../img_align_celeba/' + str(path))
    print(image)
    try:
        print('copying images...')
        if g == 1.0:
            shutil.copy('../img_align_celeba/' + str(path),
                        '../data/train/male')
        else:
            shutil.copy('../img_align_celeba/' + str(path),
                        '../data_smile/train/female')
    except:
        pass

print('done!')
