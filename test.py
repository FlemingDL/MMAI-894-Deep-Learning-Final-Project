"""Test the model"""
import argparse
import ssl
import torch
import slack
import os
import utils
import json
from torchvision import transforms, datasets
import pandas as pd
import logging
from tqdm import tqdm
from pathlib import Path
import scipy.io as spio

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")


def input_size_of_model(model_name):
    input_size = 0

    if model_name == "resnet":
        input_size = 224

    elif model_name == "alexnet":
        input_size = 224

    elif model_name == "vgg":
        input_size = 224

    elif model_name == "squeezenet":
        input_size = 224

    elif model_name == "densenet":
        input_size = 224

    elif model_name == "inception":
        input_size = 299

    elif model_name == "xception":
        input_size = 299

    else:
        logging.info("Invalid model name, exiting...")
        exit()

    return input_size


def load_checkpoint(filepath, device, inference=False):
    checkpoint = torch.load(os.path.join(filepath, 'checkpoint.pt'))
    model = checkpoint['model']
    if inference:
        for parameter in model.parameter():
            parameter.require_grad = False
        model.eval()
    return model


def post_slack_message(message):
    if 'SLACK_API_TOKEN' in os.environ:
        client.chat_postMessage(channel='CU3JRRA4E', text=message)


if __name__ == '__main__':

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
        with open(json_path) as f:
            params_text = json.load(f)
        post_slack_message("Testing model for experiment: {}\n"
                           "The parameters used in training were:{}".format(args.model_dir, params_text))

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'test.log'))

    # Set variables
    data_dir = "./data/"
    model_name = params.model_name
    batch_size = params.test_batch_size
    num_workers = params.num_workers

    # Get the required input size of the network for resizing images
    input_size = input_size_of_model(model_name)

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
    model_ft = load_checkpoint(filepath=args.model_dir, device=device)

    # Send the model to device (GPU or CPU)
    model_ft = model_ft.to(device)

    # Create test image dataset
    test_data = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=data_transforms)

    # Create test dataloaders
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

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

    # Set model to evaluate
    model_ft.eval()

    for inputs, _ in tqdm(test_data_loader):
        torch.no_grad()
        inputs = inputs.to(device)
        outputs = model_ft(inputs)
        _, pred = torch.max(outputs, 1)

        for i in range(len(inputs)):
            file_names.append(image_names[i])
            prediction = pred[i]
            predictions.append(prediction.item() + 1)
            car_names.append(cars_classid_to_name.iloc[prediction]['name'])

    # save the predictions
    predictions_file = os.path.join(args.model_dir, 'predictions_file.txt')
    df = pd.DataFrame()
    df['file_name'] = file_names
    df['predicted_class_id'] = predictions
    df['predicted_car_name'] = car_names
    df.to_csv(predictions_file, index=None)
    print('Done testing!')



