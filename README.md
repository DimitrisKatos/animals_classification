# Animals Classification -- Project in PROCESS 

In this Notebook we will make different machine learning models to classify 10 different animals. We will both use PyTorch and TensorFlow to create many models.
In this project we will try to classify animals in 10 different classes. The project contains the following:
1. **Data Preprocessing**: In this part we turn the data into a format that is more useful for our project. We also create many different dataset that contains training, testing and validation sets.

2. **Modeling with PyTorch**: We are modeling using PyTorch, creating helpful functionalities to `going_modular` folder and we track our experiments.

3. **Modeling using TensorFlow**: We use TensorFlow to create models with different hyperparameters.

4. **Deploy model**: Deploy the best model in HuggingFace Spaces.


# 1. Data Preprocessing.
This is the first step of the project, in which we have to main goals.
1. Create 3 new datataset and every one of them will contain a different number or training images.
2. Change the structure of the data, in a format that is more useful for our project. The structure is the following.


The input of this process is the original dataset that contains 10 different classes of animals. The name of the classes is in Italian, so we need to fix this. Also, the dataset hasn't testing and validation files. 

The process of this step does the following:
1. Unzip the original Dataset.
2. Rename the folders in English
3. Create testing and validation sets.
4. Create a big training set, that contains all the images except the training and validation sets.
5. Create 3 Dataset:
    * `transformed_dataset_25_images`: Contains 25 images of every class. Total training images is 250.
    * `transformed_dataset_50_images`: Contains 50 images of every class. Total training images is 500.
    * `transformed_dataset_100_images`: Contains 100 images of every class. Total training images is 2500.
6. Copy the testing and validation sets to the previous 3 datasets.

Now, the dataset is in the right format that is the following.

```
transformed_dataset_25_images/ <- overall dataset folder
    train/ <- training images
        butterfly/ <- class name as folder name
            image01.jpeg
            image02.jpeg
            ...
        cat/
            image24.jpeg
            image25.jpeg
            ...
        chicken/
            image37.jpeg
            ...
    test/ <- testing images
        butterfly/
            image101.jpeg
            image102.jpeg
            ...
        cat/
            image154.jpeg
            image155.jpeg
            ...
        chicken/
            image167.jpeg
            ...
    vali/ <- testing images
        butterfly/
            image201.jpeg
            image202.jpeg
            ...
        cat/
            image254.jpeg
            image255.jpeg
            ...
        chicken/
            image267.jpeg
            ...

```

All the datasets has the same number of training and validation images. Finally, the 4 datasets contain the same test set that will be used for the final evaluation of the models.

The processes that walk through the dataset are stored in `data_preprocessing_folders` and contains the following files:
* `data_preprocessing.ipynb`  - It is the raw file that shows the process of modifying our data to create some datasets.
* `config.py` - Stored all the hyperparameters that is used in the following files.
* `utils.py` - Consists functionalities that was created in `data_preprocessing.ipynb`. By saving the functionalities in another file helps to have more clean code and improve it or maintain bases on the necessaries of every project.
* `main.py` - In this file we call the functions we have created in the `utils.py`. 

# 2. Modeling with PyTorch.
The second step of the project contains using PyTorch to create many models that will classify the animals into categories. 
The process of modeling is split into different folders.

* `modeling.ipynb` : In this file we create our fist model by using Transfer Learning. 

* `going_modular folder`: Because we want to track experiments through many different hyperparameters and dataset, it is good idea to create python scripts that store our functionalities. This will help us to recall the code we have create so far. So, this subfolder contains the following files:
    * `data_setup.py` : Creates DataLoaders
    * `engine.py` : Contains the training step
    * `model_builder.py`: Create the model
    * `utils.py`: Useful functionalities
    * `config.py`: stored hyperparameters useful for the problem.
    * `prediction.py`: Contains fucntionallities for evaluating the model. 

* `experiment_tracking.ipynb`: In this notebook we try different hyperparameters and datasets for creating the best model we can.  

# 3. Modeling with TensorFlow.
In Process ... 


# 4. Deploy our model.
In process ... 
