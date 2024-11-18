import os
def get_class_names(train_dir):
    return [folder for folder in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, folder))]