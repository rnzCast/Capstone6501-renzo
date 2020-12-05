![logo](img/logo.jpg)



**Introduction**
---
Machine Learning (ML) is playing an important role in real-world tasks and explaining why a 
ML model made a decision could be crucial. The importance of this research 
arises because gender recognition is crucial in social interactions with intelligent 
applications. 

The scope of this project will be to use three pre-trained models (VG16, Inception, and Resnet),
to infer gender (male or female) from images, and to interpret the results with Class Activation Maps (CAMs)
and SHAP.

The data that will be used for this project is the CelebA dataset [1], which contains 200K images of 
celebrities with their corresponding attributes, including gender.

To see the full project paper please  visit:
[Link to paper](https://github.com/rnzCast/Capstone6501-renzo/blob/master/Deliverables/Final%20Submission%20-%20Journal/Journal%20-%20Renzo%20P.%20Castagnino.docx)

Considering the complexity of the project, and to the limitations in GitHub to upload the data, 
there are multiple steps and instructions to follow in order to run the code.


**Installation Options**
---

1. Install with [`pip`]
    + `$ pip install shap`
    + `$ pip install path`
    + `$ pip install torch`
    + `$ pip install torchvision==0.1.8`
    



**Configuration and Setup**
---

**1. GETTING THE DATA READY**

The first part is have the data ready to run the pre-trained models. The following steps will guide to
though the process of downloading the data and prepare it for the training.

1.1. Download the dataset:
[Link to CelebA Dataset](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing)
          
1.2. Unzip the folder and paste it in the directory of the project. The name of this folder should be 
`img_align_celeba`

1.3. create an empty `data` folder as follows:
![logo](img/data_sample.jpg)

The folder should have `test` and `train` folders, with `female` and `male` each one. It's ok if it is emtpy;
this will be populated after we run our `load_data.py` file.

1.4. Open and run the file `load_data.py`. This file will read the images from the`img_align_celeba`, 
will read the .csv file `celeba_list_attr.csv`, and will copy the images accordingly in the `data` folder
with a train/test split of 80 - 20%. After the process is done, check that the images are properly copied.

1.5. Done! Data is ready to be run for the model.

**2. RUNNING THE PRE-TRAINED MODELS**


2. Running the pre-trained models. before running the models, you need to create a folder named
  'models', where all the models will be saved after trained. Also, you need to create an empty
  folder named 'PreTrained_Models', where all the pre-trained models will be downloaded.

    + VGG16
    	- This file is ready to run as is.
    + Inception
        - This file is ready to run as is.
    + ResNet
        - This file is ready to run as is.

3. Running the Interpretability models

    + Class Activation Maps (CAMs)
        - This file will read all the images in the folder 'val_gender', and will evaluate
          the class activation map. The results will be saved in a folder 'image_results'
    + SHAP
        - This file analyzes one image at a time. you need to specify the image name in the 'img_name'
          variable, and also the path where the image is located 'folder_path'. Furthermore, you need
          to specify which pre-trained model will be used in 'model_to_run', 1 for VGG16, 2 for Inception
          and 3 for ResNet.



**References**
---

1. http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

2. 