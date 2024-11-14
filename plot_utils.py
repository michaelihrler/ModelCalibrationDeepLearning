from matplotlib import pyplot as plt

def plot_histogram_balance_of_dataset(categories, counts, title):
    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts, color='skyblue')
    plt.xlabel('Dataset Categories')
    plt.ylabel('Number of Samples')
    plt.title(title)
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
