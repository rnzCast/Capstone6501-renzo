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

The data that will be used for this project is the Celeba dataset [1], which contains 200K images of 
celebrities with their corresponding attributes, including gender.

To see the full project paper please  visit:

https://github.com/rnzCast/Capstone6501-renzo/blob/master/Deliverables/Final%20Submission%20-%20Journal/Journal%20-%20Renzo%20P.%20Castagnino.docx

The following steps contain the instructions and setup to run the code for this project.


**Installation Options**
---

1. Install with [`pip`]
    + `$ pip install shap`
    + `$ pip install path`
    + `$ pip install torch`
    + `$ pip install torchvision==0.1.8`
    



**Configuration and Setup**
---

1. The Data

There are a few steps in order to have the data ready to be analyze by the model. The first part is
to ge the data ready. You need to download the images from the Celeba Dataset website, and paste it 
our repository. All the instructions to ge the data ready for the model are specified below.

    + Download the images from the Celeba dataset link.
    
        - The folder that you need to download is the 'img_align_celeba_png.7z'. don't worry for the .csv file with the labels, you will 
          find it here in the repo. 
          https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg
          
    + Setting up the data in the repository
        - Unzip the images folder 'img_align_celeba_png' and paste it in the main repository.
        
    + Running the 'load_data.py' file to get the data ready for the model.
        - The 'load_data' file, will read the images and the .csv file, and will separate
          the images based on the gender label. You need to  create an empty 'data' folder and subfolders
          as follows:
            >data
                >test
                    >female
                    >male
                >train
                    >female
                    >male
              
          

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