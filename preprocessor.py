import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def data_loader(data_dir_normal, data_dir_aug):
    images_normal = list(data_dir_normal.glob("*.png"))
    print("Number of normal images found: ", len(images_normal))

    images_aug = list(data_dir_aug.glob("*.png"))
    print("Number of augmented images found: ", len(images_aug))

    # Get list of all the images
    images = images_normal + images_aug
    print("Number of totall images found: ", len(images))

    # Let's take a look at some samples first.
    # Always look at your data!
    sample_images = images[:4]

    _, ax = plt.subplots(2, 2, figsize=(5, 3))
    for i in range(4):
        img = cv2.imread(str(sample_images[i]))
        print("Shape of image: ", img.shape)
        ax[i // 2, i % 2].imshow(img)
        ax[i // 2, i % 2].axis('off')
    plt.show()
    return images


# Iterate over the dataset and store the information needed
def get_dataset(images):
    code_length = []  # A list to store the length of each national code
    dataset = []  # Store image-label info
    characters = set()  # Store all the characters in a set
    for img_path in images:
        # 1. Get the label associated with each image
        label = img_path.name.split(".png")[0]
        # 2. Store the length of this national code
        code_length.append(len(label))
        # 3. Store the image-label pair info
        dataset.append((str(img_path), label))
        # 4. Store the characters present
        for ch in label:
            characters.add(ch)

    # Sort the characters
    characters = sorted(characters)
    # Convert the dataset info into a dataframe
    dataset = pd.DataFrame(dataset, columns=["img_path", "label"], index=None)
    # Shuffle the dataset
    dataset = dataset.sample(frac=1.).reset_index(drop=True)
    print("Number of unqiue charcaters in the whole dataset: ", len(characters))
    print("Maximum length of any national code: ", max(Counter(code_length).keys()))
    print("Characters present: ", characters)
    print("Total number of samples in the dataset: ", len(dataset))
    dataset.head()
    return characters, code_length, dataset


# Sanity check for corrupted images
def is_valid_code(code, characters):
    for ch in code:
        if not ch in characters:
            return False
    return True


# Store arrays in memory as it's not a muvh big dataset
def generate_arrays(df, characters, resize = True, img_height = 128, img_width = 512):
    """Generates image array and labels array from a dataframe.

    Args:
        df: dataframe from which we want to read the data
        resize (bool)    : whether to resize images or not
        img_weidth (int): width of the resized images
        img_height (int): height of the resized images

    Returns:
        images (ndarray): grayscale images
        labels (ndarray): corresponding encoded labels
    """
    num_items = len(df)
    images = np.zeros((num_items, img_height, img_width), dtype=np.float32)
    labels = [0] * num_items

    for i in range(num_items):
        img = cv2.imread(df["img_path"][i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if resize:
            img = cv2.resize(img, (img_width, img_height))
        img = (img / 255.).astype(np.float32)
        label = df["label"][i]
        # Add only if it is a valid code
        if is_valid_code(label, characters):
            images[i, :, :] = img
            labels[i] = label
    return images, np.array(labels)


# Split the dataset into training and test sets
def split_data(dataset, characters):
    seed = 1234
    np.random.seed(seed)
    tf.random.set_seed(seed)
    training_data, test_data = train_test_split(dataset, test_size=0.2, random_state=seed)
    training_data = training_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    print("Number of training samples: ", len(training_data))
    print("Number of test samples: ", len(test_data))

    # Map text to numeric labels
    char_to_labels = {char: idx for idx, char in enumerate(characters)}
    # Map numeric labels to text
    labels_to_char = {val: key for key, val in char_to_labels.items()}
    # Build training data
    training_data, training_labels = generate_arrays(df=training_data, characters=characters)
    print("Number of training images: ", training_data.shape)
    print("Number of training labels: ", training_labels.shape)
    # Build test data
    test_data, test_labels = generate_arrays(df=test_data, characters=characters)
    print("Number of test images: ", test_data.shape)
    print("Number of test labels: ", test_labels.shape)
    return training_data, training_labels, test_data, test_labels, char_to_labels, labels_to_char




