# main.py
import os
from config import *
from utils import *
from tqdm.auto import tqdm

def main():
    # Set up logging
    setup_logging()

    '''# Step 1: Download dataset
    if not os.path.exists(RAW_DATA_DIR):
        download_dataset(DATASET_URL, RAW_DATA_DIR)  # Implement this function if needed.'''

    # Step 2: Unzip dataset
    unzip_data(os.path.join(BASE_DIR, "animals10.zip"), RAW_DATA_DIR, unzip_path = BASE_DIR)

    # Step 3: Rename folders
    rename_folders(RAW_DATA_DIR, TRANSLATIONS)

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
    copy_test_set(DATASET_SIZES, BASE_DIR)


    FOLDER_TO_DELETE.append('/raw-img')
    FOLDER_TO_DELETE.append('/test')

    
    delete_directory(directory_path= BASE_DIR +'/transformed_dataset_50_images')
    delete_directory(directory_path= BASE_DIR +'/transformed_dataset_100_images')
    delete_directory(directory_path= BASE_DIR +'/transformed_dataset_25_images')
    delete_directory(directory_path= BASE_DIR +'/raw-img')
    delete_directory(directory_path= BASE_DIR +'/test')
if __name__ == "__main__":
    main()
