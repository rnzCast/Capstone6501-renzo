import pandas as pd
import cv2
import shutil
from pathlib import Path

DATA_DIR = (str(Path(__file__).parents[1]) + '/data/')
df = pd.read_csv(DATA_DIR + 'imdb/imdb.csv')

for i in df.values:
    g = i[3]  # Gender
    i = i[2]  # Path
    path = i.split('/')
    path = path[1]
    image = cv2.imread(DATA_DIR + 'img/' + path)

    try:
        print('copying images...')
        if g == 0.0:
            shutil.copy(DATA_DIR + 'img/' + path,
                        DATA_DIR + 'data/train/female')
        else:
            shutil.copy(DATA_DIR + 'img/' + path,
                        DATA_DIR + 'data/train/male')
    except:
        pass

print('done!')
