"""Test the model"""
import argparse
import ssl
import torch
import slack
import os
import utils
from torchvision import transforms, datasets
import pandas as pd
import logging
from tqdm import tqdm
from pathlib import Path
import scipy.io as spio
import time
import model_handler as mh

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")


# def input_size_of_model(model_name):
#     input_size = 0
#
#     if model_name == "resnet":
#         input_size = 224
#
#     elif model_name == "resnet152":
#         input_size = 224
#
#     elif model_name == "alexnet":
#         input_size = 224
#
#     elif model_name == "vgg":
#         input_size = 224
#
#     elif model_name == "squeezenet":
#         input_size = 224
#
#     elif model_name == "densenet":
#         input_size = 224
#
#     elif model_name == "inception":
#         input_size = 299
#
#     elif model_name == "xception":
#         input_size = 299
#
#     elif model_name == "fleming":
#         input_size = 224
#
#     else:
#         logging.info("Invalid model name, exiting...")
#         exit()
#
#     return input_size


# def load_checkpoint(filepath, device):
#     checkpoint = torch.load(os.path.join(filepath, 'checkpoint.pt'), map_location=str(device))
#     torch.load
#     model = checkpoint['model']
#     return model


def post_slack_message(message, response=None):
    if 'SLACK_API_TOKEN' in os.environ:
        if response is None:
            try:
                response = client.chat_postMessage(channel=slack_channel, text=message)
            except:
                logging.info('Error posting message to slack')
        else:
            try:
                response = client.chat_update(channel=response['channel'], ts=response['ts'], text=message)
            except:
                logging.info('Error posting message to slack')

        return response


def post_slack_file(file_name, response=None):
    if 'SLACK_API_TOKEN' in os.environ:
        if response is None:
            try:
                response = client.files_upload(channels=slack_channel, file=file_name, filename=file_name)
            except:
                logging.info('Error uploading file to slack')
        else:
            delete_slack_file(response)
            try:
                response = client.files_upload(channels=slack_channel, file=file_name, filename=file_name)
            except:
                logging.info('Error uploading file to slack')

        return response


def delete_slack_message(response):
    if 'SLACK_API_TOKEN' in os.environ:
        try:
            client.chat_delete(channel=response['channel'], ts=response['ts'])
        except:
            logging.info('Error deleting message')


def delete_slack_file(response):
    if 'SLACK_API_TOKEN' in os.environ:
        client.files_delete(file=response['file']['id'])


if __name__ == '__main__':

    # Set slack channel
    slack_channel = '#dl-model-progress'
    # slack_channel = '#temp'

    # Collect arguments from command-line options
    args = parser.parse_args()

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    if 'SLACK_API_TOKEN' in os.environ:
        # Setup slack messages to track progress
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        client = slack.WebClient(token=os.environ['SLACK_API_TOKEN'], ssl=ssl_context)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'test.log'))

    slack_message = "*Testing of {} started*".format(args.model_dir)
    post_slack_message(slack_message)

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
    post_slack_message(message)

    # save the predictions
    predictions_file = os.path.join(args.model_dir, 'file_name_prediction.txt')
    submission_file = os.path.join(args.model_dir, 'submit_to_stanford_prediction.txt')

    df = pd.DataFrame()
    df['file_name'] = file_names
    df['predicted_class_id'] = predictions
    df['predicted_car_name'] = car_names

    df[['file_name', 'predicted_class_id', 'predicted_car_name']].to_csv(predictions_file, index=None)
    df['predicted_class_id'].to_csv(submission_file, index=None, header=None)

    post_slack_file(predictions_file)
    post_slack_file(submission_file)
    logging.info('Done testing!')
