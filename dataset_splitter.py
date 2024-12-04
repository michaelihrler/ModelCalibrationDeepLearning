import os
import shutil
import random


def copy_n_random_files(n,src_dir, dest_dir):
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
