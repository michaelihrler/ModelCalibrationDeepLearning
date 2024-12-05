import os
import random
import shutil

def copy_n_random_files(n, src_dir, train_dir, val_dir, train_val_ratio=0.8):
    """
    Copies n random files from the source directory to the train and validation directories
    based on the specified train-validation ratio.

    Parameters:
        n (int): Total number of files to copy.
        src_dir (str): Source directory containing the files.
        train_dir (str): Destination directory for training files.
        val_dir (str): Destination directory for validation files.
        train_val_ratio (float): Ratio of files to be copied to the train directory (default is 0.8).
    """
    # Get all files from the source directory
    all_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

    # Check if n is greater than the number of available files
    if n > len(all_files):
        raise Exception(f"n={n} is greater than the number of available files ({len(all_files)}) in {src_dir}.")

    # Randomly select n unique files
    selected_files = random.sample(all_files, n)

    # Split files into train and validation sets
    train_count = int(n * train_val_ratio)
    val_count = n - train_count
    train_files = selected_files[:train_count]
    val_files = selected_files[train_count:]

    # Ensure the destination directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Copy files to the train directory
    for file_name in train_files:
        src_file_path = os.path.join(src_dir, file_name)
        dest_file_path = os.path.join(train_dir, file_name)
        shutil.copy(src_file_path, dest_file_path)

    # Copy files to the validation directory
    for file_name in val_files:
        src_file_path = os.path.join(src_dir, file_name)
        dest_file_path = os.path.join(val_dir, file_name)
        shutil.copy(src_file_path, dest_file_path)

    print(f"Copied {train_count} files to {train_dir} and {val_count} files to {val_dir}.")
