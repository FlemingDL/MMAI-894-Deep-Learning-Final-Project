"""Download the Cars Dataset from https://ai.stanford.edu/~jkrause/cars/car_dataset.html
---
The Cars Dataset comes as a training set 'cars_train.tgz', a test set 'cars_test.tgz', and
a devkit that includes class labels for training images and bounding boxes for all images.

Example:
    python download_data.py
---
"""

import ssl

import wget

# Run the following when run as the main module
if __name__ == '__main__':

    # Ignore ssl certification (prevent error for some users)
    ssl._create_default_https_context = ssl._create_unverified_context

    print('Downloading training images...')
    wget.download('http://imagenet.stanford.edu/internal/car196/cars_train.tgz')

    print('\nDownloading test images...')
    wget.download('http://imagenet.stanford.edu/internal/car196/cars_test.tgz')

    print('\nDownloading devkit (image labels and bounding boxes)...')
    wget.download('https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz')

    print('Done downloading!')
