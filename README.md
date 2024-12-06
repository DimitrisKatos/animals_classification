# Animals Classification -- Porject in PROCESS 

In this Notebook we will make different machine learning models to classify 10 different animals. We will both use PyTorch and TensorFlow to create many models.
In this project we will try to classify animals in 10 different classes. The project contains the following:
1. Data Preprocessing: In this part we turn the data into a format that is more useful for our project. We also create many different dataset that contains training, testing and validation sets.

2. Modeling with PyTorch: We our modeling using PyTorch, and we track our experiments.

3. Modeling using TensorFlow: We use TensorFlow to create models with different hyperparameters.

4. Deploy model: Deploy the best model in HuggingFace Spaces.


# 1. Data Preprocessing.
This is the first step of the project, in which we modify by creating training, validation and testing datasets.

The input of this process is a dataset that contains 10 different classes of animals. The name of the classes is in Italian. Also, the dataset hasn't testing and validation files. 

The output of the process will be 4 datasets:
- `transformed_dataset_25_images`: The dataset consists 10 folder (each for an animal) and for every animal two more folders (train, valid). Both training and validation sets consists 25 images (total train images 250).
- `transformed_dataset_50_images`: The same as previous but now we have 50 images per subsets (train,valid) and per animal. (total train images 500)
- `transformed_dataset_100_images`: The same as previous but now we have 100 images per subsets (train,valid) and per animal. (total train images 1000)
- `transformed_dataset_all_images`: Contains all the images for as a training set. 

All the datasets has the same number of training and validation images. Finally, the 4 datasets contain the same test set that will be used for the final evaluation of the models.

The processes that walk through the dataset stored in `data_preprocessing_folders` and contains the following files:
* `data_preprocessing.ipynb`  - It is the raw file that shows the process of modifying our data to create some datasets.
* `config.py` - Stored all the hyperparameters that is used in the following files.
* `utils.py` - Consists functionalities that was created in `data_preprocessing.ipynb`. By saving the functionalities in another file helps to have more clean code and improve it or maintain bases on the necessaries of every project.
* `main.py` - In this file we call the functions we have created int the `utils.py`. 

# 2. Modeling with PyTorch.
The second step of the project contains usinng PyTorch to create many models that will classify the animals into categories. 
The project is stored in pytorch_folder.
* `modeling.ipynb` : The walk through for our first experience in modeling.
* going_modular folder: Because we want to track experiments through many different hyperparameters and dataset, it is good to store some file into `.py` format to make our code more easy to recall. So, this subfolder contains the following files:
    * `data_setup.py` : Creates DataLoaders
    * `engine.py` : Contains the training step
    * `model_builder.py`: Create the model
    * `utils.py`: Useful functionalities
    * `config.py`: stored hyperparameters useful for the problem.
    * `prediction.py`: Contains fucntionallities for evaluating the model. 

* `experiment_tracking.ipynb`: In this notebook we try different hyperparameters and datasets for creating the best model we can.  This is still in Process.

# 3. Modeling with TensorFlow.
In Process ... 


# 4. Deploy our model.
