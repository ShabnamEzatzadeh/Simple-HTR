from preprocessor import data_loader, get_dataset, split_data
from dataGenarator import DataGenerator
from model import build_model
from train import train
from prediction import prediction, prediction_saved_model
from pathlib import Path
import argparse
import tensorflow as tf
print("Tensorflow version: ", tf.__version__)


def main():
    # Path to the data directory
    data_dir_normal = Path("../data/normal")
    data_dir_aug = Path("../data/augmented")

    # Desired image dimensions
    img_width = 512
    img_height = 128
    batch_size = 16  # Batch size for training and test
    downsample_factor = 4  # Factor  by which the image is going to be downsampled by the convolutional blocks
    max_length = 10  # Maximum length of any code in the data

    images = data_loader(data_dir_normal, data_dir_aug)
    characters, code_length, dataset = get_dataset(images)
    training_data, training_labels, test_data, test_labels, char_to_labels, labels_to_char = split_data(dataset,
                                                                                                        characters)

    # Get a generator object for the training data
    train_data_generator = DataGenerator(data=training_data,
                                         labels=training_labels,
                                         char_map=char_to_labels,
                                         characters=characters,
                                         batch_size=batch_size,
                                         img_width=img_width,
                                         img_height=img_height,
                                         downsample_factor=downsample_factor,
                                         max_length=max_length,
                                         shuffle=True
                                         )

    # Get a generator object for the test data
    test_data_generator = DataGenerator(data=test_data,
                                        labels=test_labels,
                                        char_map=char_to_labels,
                                        characters=characters,
                                        batch_size=batch_size,
                                        img_width=img_width,
                                        img_height=img_height,
                                        downsample_factor=downsample_factor,
                                        max_length=max_length,
                                        shuffle=False
                                        )
    print("Number of train_data_generator: ", len(train_data_generator))
    print("Number of test_data_generator: ", len(test_data_generator))

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
    args = parser.parse_args()

    if args.mode in ['train', 'validate']:
        model = build_model(img_width, img_height, max_length, characters)
        print(model.summary())

        if args.mode == 'train':
            train(model, train_data_generator)
        elif args.mode == 'validate':
            prediction(model, test_data_generator, test_labels, labels_to_char, characters)

    elif args.mode == 'infer':
        prediction_saved_model(test_data_generator, test_labels, labels_to_char, characters)


if __name__ == '__main__':
    main()
