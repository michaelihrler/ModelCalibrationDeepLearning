import os

from matplotlib import pyplot as plt

from data_utils import get_class_names


def plot_histogram_balance_of_dataset(data_dir, title):
    classes = get_class_names(data_dir)

    # Count the number of files in each class directory
    class_counts = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in classes}

    # Sort classes alphabetically or by frequency for consistency
    classes_sorted = sorted(class_counts.keys())
    counts_sorted = [class_counts[cls] for cls in classes_sorted]

    # Plot the histogram
    plt.bar(classes_sorted, counts_sorted, tick_label=[f"Class {cls}" for cls in classes_sorted])
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.xticks(rotation=45, ha='right')  # Rotate class names for better visibility if many
    plt.show()


def plot_loss(train_loss, val_loss):
    epochs = range(1, len(train_loss) + 1)  # Epoch numbers

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
