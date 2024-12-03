# Animals Classification -- IN Process

In this Notebook we will make different machine learning models to classify 10 different animals. We will both use PyTorch and TensorFlow to create many models.
# 1. Data Preprocessing.
The data preprocessing phase of the project is stored in the `data_preprocessing_folder`. The folder consists 4 files.
* `data_preprocessing.ipynb`  - It is the raw file that shows how to modify our data to create some
* `config.py` - Has some hyperparameters that is used in the next files.
* `utils.py` - Consists functionalities that create in `data_preprocessin.ipynb`. It helps us to have more clean code and improve it or maintain bases on the necessaries of every project.
* `main.py` - Main functions that executes some of the utils functions.

The first step of the project consist a data manipulation. In detail, we change the format of the data in a way we want. Also we create 3 different datasets:
- `transformed_dataset_25_images`: The dataset consists 10 folder (each for an animal) and for every animal two more folders (train, valid). Both training and validation sets consists 25 images (total train images 250).
- `transformed_dataset_50_images`: The same as previous but now we have 50 images per subsets (train,valid) and per animal. (total train images 500)
- `transformed_dataset_100_images`: The same as previous but now we have 100 images per subsets (train,valid) and per animal. (total train images 100)

Of course, every dataset consists the same test set. In the test subset we have 50 images per animal.

# 2. Modeling with PyTorch.
In Process....


# 3. Modeling with TensorFlow.
In Process ... 
