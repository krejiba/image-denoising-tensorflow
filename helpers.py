import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
from PIL import Image

# Sets the seed of the random number generator
np.random.seed(10)


def load_data(path, metadata_path, img_size, n_samples, test_ratio=.2):
    """
    Loads the dataset and splits it into training and test samples

            Parameters:
                    path (str): path of directory containing the images
                    metadata_path (str): path of file containing the metadata
                    img_size (tuple of int): desired size of the image
                    n_samples (int): number of samples to load
                    test_ratio (float): percentage of samples reserved for testing (default 20%)

            Returns:
                    (x_train, y_train) (tuple of numpy arrays): images for training with the corresponding labels
                    (x_test, y_test) (tuple of numpy arrays): images for testing with the corresponding labels
    """
    img_files = glob(path+'*.png')
    metadata_df = pd.read_csv(metadata_path)

    # Load only a subset of the dataset
    img_files_subset = np.random.choice(img_files, size=n_samples, replace=False)

    data = []
    labels = []

    for file_path in img_files_subset:

        # Fetch image
        image = Image.open(file_path).convert('L')
        # Resize original image
        image = image.resize(img_size)

        image_array = np.array(image)
        data.append(image_array)

        # Fetch label
        file_name = file_path.split(os.sep)[-1]

        label = metadata_df[metadata_df['Image Index'] == file_name]["Finding Labels"].item()
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    assert len(data) == len(labels), f'Not the same number of images and labels!'

    # Split dataset
    indices = np.random.permutation(len(data))  # Shuffle before splitting
    data = data[indices]
    labels = labels[indices]

    test_samples = int(len(data) * test_ratio)

    x_train, y_train = data[test_samples:], labels[test_samples:]
    x_test, y_test = data[:test_samples], labels[:test_samples]

    return (x_train, y_train), (x_test, y_test)


def preprocess(images):
    """
    Preprocess images.

            Parameters:
                    images (numpy array): training / test images

            Returns:
                    x (numpy array): images ready to be processed by the network
    """

    # Min-max normalization
    x = (images.astype('float') - images.min()) / images.max()

    num_channels = 1
    _, img_rows, img_cols = images.shape

    # Reshape to have appropriate dimensions for the network
    x = x.reshape([-1, img_rows, img_cols, num_channels])

    return x


def plot_dataset(images, noisy_images, n_samples=5):
    """
    Plots samples from the dataset.

            Parameters:
                    images (numpy array): images from dataset
                    noisy_images (numpy array): corrupted images
                    n_samples (int): number of samples to plot

            Returns:
                    None
    """

    rows, columns = 2, n_samples
    fig, ax = plt.subplots(rows, columns, figsize=(16, 9), sharex=True, sharey=True)

    # Choose some samples to plot
    indices = np.random.choice(np.arange(len(images)), n_samples, replace=False)

    for c in range(columns):
        ax[0, c].imshow(np.squeeze(images[indices[c]]), cmap='bone')
        ax[1, c].imshow(np.squeeze(noisy_images[indices[c]]), cmap='bone')

    plt.show()


def plot_training_curves(h):
    """
    Plots the training curves.

            Parameters:
                    h (tf.keras.callbacks.History): history object containing training and validation losses

            Returns:
                    None
    """
    loss = h.history['loss']
    val_loss = h.history['val_loss']
    num_epochs = h.params['epochs']

    plt.figure(figsize=(16, 9))
    plt.plot(np.arange(1, num_epochs + 1), loss)
    plt.plot(np.arange(1, num_epochs + 1), val_loss)
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.title('Training curves')
    plt.xlabel('Epoch')
    plt.grid()
    plt.show()


def plot_results(x_test, x_test_noisy, autoencoder_pred, n_samples=5):
    """
    Plots the predictions of the autoencoder.

            Parameters:
                    x_test (numpy array): test images
                    x_test_noisy (numpy array): corrupted test images
                    autoencoder_pred (): predictions of the autoencoder
                    n_samples (int): number of samples to plot

            Returns:
                    None
    """

    rows, columns = 3, n_samples
    fig, ax = plt.subplots(rows, columns, figsize=(16, 9), sharex=True, sharey=True)

    # Choose some samples to plot
    indices = np.random.choice(np.arange(len(autoencoder_pred)), n_samples, replace=False)

    for c in range(columns):
        ax[0, c].imshow(np.squeeze(x_test[indices[c]]), cmap='bone')
        ax[1, c].imshow(np.squeeze(x_test_noisy[indices[c]]), cmap='bone')
        ax[2, c].imshow(np.squeeze(autoencoder_pred[indices[c]]), cmap='bone')

    plt.show()
