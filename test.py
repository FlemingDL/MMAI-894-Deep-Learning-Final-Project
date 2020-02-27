"""Test the pretrained model against the test set provided by Stanford Cars Dataset.
---
The test data provided by
Stanford Cars Dataset is not labeled. This code will make predictions on the test set using the pretrained model
and output the results that are formatted for submission to Stanford.

Example:
    python test.py --model_dir <directory of experiment with checkpoint file>

    if you have a slack api token (check the channel setting in code):
    SLACK_API_TOKEN='place token here' python test.py --model_dir <directory of experiment with checkpoint file>
---
"""
import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
import scipy.io as spio
import torch
from torchvision import transforms, datasets
from tqdm import tqdm

import model_handler as mh
import utils
from slack_manager import SlackManager

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")

if __name__ == '__main__':

    # Setup Slack
    # sm = SlackManager(channel='#temp')
    sm = SlackManager(channel='#dl-model-progress')
    if 'SLACK_API_TOKEN' in os.environ:
        sm.setup(slack_api_token=os.environ['SLACK_API_TOKEN'])

    # Collect arguments from command-line options
    args = parser.parse_args()

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'test.log'))

    slack_message = "*Testing of {} started*".format(args.model_dir)
    sm.post_slack_message(slack_message)

    # Set variables
    data_dir = "./data/"
    model_name = params.model_name
    batch_size = params.test_batch_size
    num_workers = params.num_workers

    # Get the required input size of the network for resizing images
    input_size = mh.input_size_of_model(model_name)

    # Data augmentation and normalization for testing
    data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model
    logging.info('Loading saved model')
    model_ft = mh.load_checkpoint(filepath=args.model_dir, device=device)

    # Send the model to device (GPU or CPU)
    model_ft = model_ft.to(device)

    # Create test image dataset
    test_data = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=data_transforms)

    # Create test dataloaders
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    # Set model to evaluate
    model_ft.eval()

    # Get the car name lookup table
    devkit = 'devkit'
    cars_meta_data = spio.loadmat(os.path.join(devkit, 'cars_meta.mat'))
    cars_classid_to_name = [c for c in cars_meta_data['class_names'][0]]
    cars_classid_to_name = pd.DataFrame(cars_classid_to_name, columns=['name'])

    # Get the file names of images
    image_names = []
    for index in test_data_loader.dataset.imgs:
        image_names.append(Path(index[0]).stem)

    # Initialize lists
    file_names = []
    predictions = []
    car_names = []

    logging.info('Predicting cars')
    since = time.time()
    img_name_index = 0
    for inputs, _ in tqdm(test_data_loader):

        torch.no_grad()
        inputs = inputs.to(device)
        outputs = model_ft(inputs)
        _, pred = torch.max(outputs, 1)

        for i in range(len(inputs)):
            file_names.append(image_names[img_name_index])
            prediction = int(pred[i])
            predictions.append(prediction + 1)
            car_names.append(cars_classid_to_name.iloc[prediction]['name'])
            img_name_index += 1

    time_elapsed = time.time() - since
    message = 'Prediction complete in {:.0f}s on device:{}'.format(time_elapsed, str(device))
    logging.info(message)
    sm.post_slack_message(message)

    # save the predictions
    predictions_file = os.path.join(args.model_dir, 'file_name_prediction.txt')
    submission_file = os.path.join(args.model_dir, 'submit_to_stanford_prediction.txt')

    df = pd.DataFrame()
    df['file_name'] = file_names
    df['predicted_class_id'] = predictions
    df['predicted_car_name'] = car_names

    df[['file_name', 'predicted_class_id', 'predicted_car_name']].to_csv(predictions_file, index=None)
    df['predicted_class_id'].to_csv(submission_file, index=None, header=None)

    sm.post_slack_file(predictions_file)
    sm.post_slack_file(submission_file)
    logging.info('Done testing!')
