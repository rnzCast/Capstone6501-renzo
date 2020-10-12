import pandas as pd
import cv2
import shutil
from pathlib import Path

DATA_DIR = (str(Path(__file__).parents[1]) + '/data/')

df = pd.read_csv(DATA_DIR + 'imdb/imdb.csv')
print(df)

for i in df.values:
    g = i[3]
    i = i[2]
    path = i.split('/')
    path = path[1]
    image = cv2.imread(DATA_DIR + '/img/' + path)
    try:
        image = image[:, :, ::-1]
        print(image)
        print(path)
        print("success")
        if g == 0.0:
            shutil.copy(DATA_DIR + '/img/' + path,
                        DATA_DIR + '/train/img/female')
        else:
            shutil.copy(DATA_DIR + '/img/' + path,
                        DATA_DIR + '/train/img/male')
    except:
        pass
