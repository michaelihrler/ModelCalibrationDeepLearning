import os
import shutil
import random


def copy_n_random_files(src_dir, dest_dir, n):
    # Get all files from the source directory
    all_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

    # If n is greater than the number of available files
    if n> len(all_files) :  raise Exception("n="+n  +" bigger than actual files in " + src_dir)

    # Randomly select n unique files
    selected_files = random.sample(all_files, n)

    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Copy each selected file to the destination directory
    for file_name in selected_files:
        src_file_path = os.path.join(src_dir, file_name)
        dest_file_path = os.path.join(dest_dir, file_name)
        shutil.copy(src_file_path, dest_file_path)

    print(f"Copied {n} files from {src_dir} to {dest_dir}.")


def split_data_set_binary_classification(n, src_dir, dest_dir):
    # Delete dest_dir if it exists
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    folders = [name for name in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, name))]
    for folder in folders:
        dest_folder = os.path.join(dest_dir, folder)
        os.makedirs(os.path.join(dest_dir, folder), exist_ok=True)
        copy_n_random_files(os.path.join(src_dir, folder), dest_folder, n)


split_data_set_binary_classification( 400, "chest_xray/train", "train_data_chest_xray_balanced1")
#split_data_set_binary_classification(128, "chest_xray/test", "test_data_chest_xray_3_3_split")
