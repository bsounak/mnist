"""mnist util functions"""
import numpy as np
import os
import subprocess
import urllib.request
import gzip
from matplotlib import pyplot as plt

cache = os.path.join(os.path.expanduser("~"), ".mnist")
if not os.path.exists(cache):
    os.makedirs(cache)


def download():
    """
    Download mnist database
    """
    urllib.request.urlretrieve(
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        os.path.join(cache, "train-images-idx3-ubyte.gz"),
    )
    urllib.request.urlretrieve(
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        os.path.join(cache, "train-labels-idx1-ubyte.gz"),
    )
    urllib.request.urlretrieve(
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        os.path.join(cache, "t10k-images-idx3-ubyte.gz"),
    )
    urllib.request.urlretrieve(
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
        os.path.join(cache, "t10k-labels-idx1-ubyte.gz"),
    )


def load():
    """
    Load mnist data

    Args:

    Returns:
        training_set_image (numpy.ndarray): shape (60000, 28, 28)
        training_set_label (numpy.ndarray): shape (60000,)
        test_set_image (numpy.ndarray): shape (10000, 28, 28)
        test_set_label (numpy.ndarray): shape (10000,)
    """
    try:
        with gzip.open(os.path.join(cache, "train-images-idx3-ubyte.gz"), "rb") as f:
            magic_number = f.read(4)
            number_of_images = int.from_bytes(f.read(4), byteorder="big")
            nrow = int.from_bytes(f.read(4), byteorder="big")
            ncol = int.from_bytes(f.read(4), byteorder="big")

            img_data = np.frombuffer(f.read(), dtype=np.uint8)
            training_set_image = img_data.reshape(number_of_images, nrow, ncol)

        with gzip.open(os.path.join(cache, "train-labels-idx1-ubyte.gz"), "rb") as f:
            magic_number = f.read(4)
            number_of_items = int.from_bytes(f.read(4), byteorder="big")
            training_set_label = np.frombuffer(f.read(), dtype=np.uint8)

        with gzip.open(os.path.join(cache, "t10k-images-idx3-ubyte.gz"), "rb") as f:
            magic_number = f.read(4)
            number_of_images = int.from_bytes(f.read(4), byteorder="big")
            nrow = int.from_bytes(f.read(4), byteorder="big")
            ncol = int.from_bytes(f.read(4), byteorder="big")

            img_data = np.frombuffer(f.read(), dtype=np.uint8)
            test_set_image = img_data.reshape(number_of_images, nrow, ncol)

        with gzip.open(os.path.join(cache, "t10k-labels-idx1-ubyte.gz"), "rb") as f:
            magic_number = f.read(4)
            number_of_items = int.from_bytes(f.read(4), byteorder="big")

            test_set_label = np.frombuffer(f.read(), dtype=np.uint8)

    except FileNotFoundError as err:
        raise Exception(
            "mnist database seems to be missing. "
            "make sure you have called mnist.download() first"
        ) from err

    return training_set_image, training_set_label, test_set_image, test_set_label


def preview():
    """
    Copied this function from a cs231n assignment notebook

    Visualize some examples from the dataset.  We show a few examples of
    training images from each class.
    """
    X_train, y_train, X_test, y_test = load()
    classes = np.unique(y_train)
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype("uint8"))
            plt.axis("off")
            if i == 0:
                plt.title(cls)
    plt.show()
