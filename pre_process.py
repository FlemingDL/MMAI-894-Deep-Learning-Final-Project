"""Split the cars dataset into train/val/test
---
The cars dataset comes in the following format:
    cars_train/
        00001.jpg
        00002.jpg
        ...
    cars_test/
        00001.jpg
        00002.jpg
        ...

The Stanford Cars Dataset "cars_train" data needs to be split into training (train) and validation (val) sets.  The
"cars_test" data has no labels.  The provided test set is for submitting predictions to the Stanford Cars database
to determine the accuracy of the model (you can only supply one text file every 24 hours).

Further image preprocessing, such as resizing, cropping and normalization is done in the train.py and test.py

The ImageFolder class provided by pytorch handles the class label of each image, providing the images are in a
structure where each each image is saved in a folder where the folder name is the label of the images in that folder.

The cars dataset also provides bounding boxes around the cars the image.

Example:
    python pre_process.py --model_dir <path to experiment folder that contains params.json>
---
"""

import argparse
import logging
import os
import random
import shutil
import tarfile

import cv2 as cv
import numpy as np
import scipy.io
from tqdm import tqdm

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_annotations(file_name):
    src_path = os.path.join('devkit', file_name)
    cars_annotations = scipy.io.loadmat(src_path)
    annotations = cars_annotations['annotations']
    annotations = np.transpose(annotations)
    return annotations


def get_bounding_box(annotation):
    bbox_x1 = annotation[0][0][0][0]
    bbox_y1 = annotation[0][1][0][0]
    bbox_x2 = annotation[0][2][0][0]
    bbox_y2 = annotation[0][3][0][0]
    return bbox_x1, bbox_y1, bbox_x2, bbox_y2


def process_train_data(params):
    logging.info('Processing training data. Crop to bounding box set to: {}'.format(params.crop_images_to_bounding_box))
    annotations = get_annotations('cars_train_annos')

    bounding_boxes = []
    file_names = []
    class_ids = []
    labels = []

    for annotation in annotations:
        bounding_boxes.append((get_bounding_box(annotation)))
        class_id = annotation[0][4][0][0]
        labels.append('%04d' % (class_id,))
        file_name = annotation[0][5][0]
        file_names.append(file_name)
        class_ids.append(class_id)

    labels_count = np.unique(class_ids).shape[0]
    logging.info('The number of different cars is %d' % labels_count)

    save_train_data(file_names, labels, bounding_boxes, params)


def crop_and_save_image(src_folder, file_name, bounding_box, dst_path):
    (x1, y1, x2, y2) = bounding_box
    src_path = os.path.join(src_folder, file_name)
    src_image = cv.imread(src_path)
    height, width = src_image.shape[:2]
    # margins of 16 pixels
    margin = 16
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(x2 + margin, width)
    y2 = min(y2 + margin, height)
    cropped_image = src_image[y1:y2, x1:x2]
    cv.imwrite(dst_path, cropped_image)


def save_image(src_folder, file_name, dst_path):
    src_path = os.path.join(src_folder, file_name)
    src_image = cv.imread(src_path)
    cv.imwrite(dst_path, src_image)


def save_train_data(filenames, labels, bounding_boxes, params):
    src_folder = 'cars_train'
    num_samples = len(filenames)

    train_split = params.train_data_split
    num_train = int(round(num_samples * train_split))
    train_indexes = random.sample(range(num_samples), num_train)

    for i in tqdm(range(num_samples)):
        file_name = filenames[i]
        label = labels[i]
        bounding_box = bounding_boxes[i]

        if i in train_indexes:
            dst_folder = 'data/train'
        else:
            dst_folder = 'data/val'

        # Save each image in a folder where the folder name is the label of the image.
        # The ImageFolder class used in train.py handles the class label of each image
        # (providing the images are in a structure where each directory is a label)
        dst_path = os.path.join(dst_folder, label)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst_path = os.path.join(dst_path, file_name)

        if params.crop_images_to_bounding_box:
            crop_and_save_image(src_folder, file_name, bounding_box, dst_path)
        else:
            save_image(src_folder, file_name, dst_path)


def process_test_data(params):
    logging.info('Processing test data. Crop to bounding box set to: {}'.format(params.crop_images_to_bounding_box))
    annotations = get_annotations('cars_test_annos')

    bounding_boxes = []
    file_names = []

    for annotation in annotations:
        bounding_boxes.append((get_bounding_box(annotation)))
        file_name = annotation[0][4][0]
        file_names.append(file_name)

    save_test_data(file_names, bounding_boxes)


def save_test_data(file_names, bounding_boxes):
    src_folder = 'cars_test'
    dst_folder = 'data/test/no_label'
    num_samples = len(file_names)

    for i in tqdm(range(num_samples)):
        file_name = file_names[i]
        bounding_box = bounding_boxes[i]
        dst_path = os.path.join(dst_folder, file_name)

        if params.crop_images_to_bounding_box:
            crop_and_save_image(src_folder, file_name, bounding_box, dst_path)
        else:
            save_image(src_folder, file_name, dst_path)


if __name__ == '__main__':

    # Collect arguments from command-line options
    args = parser.parse_args()

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'pre_process.log'))

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Extract the zip files if they haven't been extracted
    logging.info('Checking training set...')
    if not os.path.exists('cars_train'):
        logging.info('No cars_train folder.  Extracting cars_train.tgz')
        with tarfile.open('cars_train.tgz', "r:gz") as tar:
            tar.extractall()

    logging.info('Checking testing set...')
    if not os.path.exists('cars_test'):
        logging.info('No cars_test folder.  Extracting cars_test.tgz')
        with tarfile.open('cars_test.tgz', "r:gz") as tar:
            tar.extractall()

    logging.info('Checking devkit set...')
    if not os.path.exists('devkit'):
        logging.info('No cars_train folder.  Extracting cars_train.tgz')
        with tarfile.open('car_devkit.tgz', "r:gz") as tar:
            tar.extractall()

    # Get the cars meta data
    cars_meta_data = scipy.io.loadmat('devkit/cars_meta')
    # Get class names
    class_names = cars_meta_data['class_names']  # the shape is 1 x 196
    class_names = np.transpose(class_names)  # convert to 196 x 1
    logging.info('Class names loaded...')
    logging.info('Examples of class names: [{}]'.format(class_names[8][0][0]))

    # remove any previous data
    if os.path.exists('data'):
        logging.info('Remove previous files...')
        shutil.rmtree('data')

    ensure_folder('data/train')
    ensure_folder('data/val')
    ensure_folder('data/test/no_label')

    process_train_data(params)
    process_test_data(params)

    # clean up
    logging.info('Cleaning up...')
    shutil.rmtree('cars_train')
    shutil.rmtree('cars_test')

