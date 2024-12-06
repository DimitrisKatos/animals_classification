# main.py
import os
from config import *
from utils import *
from tqdm.auto import tqdm

def main():
    # Set up logging
    setup_logging()

    # Step 1: Unzip dataset
    unzip_data(os.path.join(BASE_DIR, "animals10.zip"), RAW_DATA_DIR, unzip_path = BASE_DIR)

    # Step 2: Rename folders
    rename_folders(RAW_DATA_DIR, TRANSLATIONS)

    # Step 3: Create big training set and a validation.
    create_test_set(os.path.join(BASE_DIR, "raw-img"),
                     os.path.join(BASE_DIR, '/transformed_dataset_all_images/train'),
                     classes= CLASSES)

    # Step 4: Create transformed datasets
    for dataset_name, num_images in DATASET_SIZES.items():
        dataset_path = os.path.join(BASE_DIR, dataset_name)
        for subset in SETS:
            for animal in TRANSLATIONS.values():
                source = os.path.join(RAW_DATA_DIR, animal)
                dest = os.path.join(dataset_path, subset, animal)
                copy_images(source, dest, num_images= num_images)

    # Step 5: Create the test set and copy it to every dataset.
    create_test_set(RAW_DATA_DIR, os.path.join(BASE_DIR, 'test'), classes= CLASSES)

    DATASET_SIZES['tranformed_dataset_all_images'] = None
    copy_test_set(DATASET_SIZES, BASE_DIR)
    
    delete_directory(directory_path= BASE_DIR +'/transformed_dataset_50_images')
    delete_directory(directory_path= BASE_DIR +'/transformed_dataset_100_images')
    delete_directory(directory_path= BASE_DIR +'/transformed_dataset_25_images')
    delete_directory(directory_path= BASE_DIR + '/tranformed_dataset_all_images')
    delete_directory(directory_path= BASE_DIR +'/raw-img')
    delete_directory(directory_path= BASE_DIR +'/test')
if __name__ == "__main__":
    main()
