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

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

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

    else:
        logging.info("Invalid model name, exiting...")
        exit()

    return input_size


def load_checkpoint(filepath, inference=False):
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
    batch_size = params.batch_size
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

    # Load the model
    model_ft = load_checkpoint(filepath=args.model_dir)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to device (GPU or CPU)
    model_ft = model_ft.to(device)

    # Create test image dataset
    test_data = datasets.ImageFolderWithPaths(root=os.path.join(data_dir, 'test'), transform=data_transforms)

    # Create test dataloaders
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    # Set model to evaluate
    model_ft.eval()

    image_paths = []
    predictions = []

    for inputs, _, paths in test_data_loader:
        torch.no_grad()
        inputs = inputs.to(device)
        outputs = model_ft(inputs)
        _, pred = torch.max(outputs, 1)

        print('image paths: {}'.format(paths))
        image_paths.append(paths)
        print('prediction: {}'.format(pred))
        predictions.append(pred)

    # TODO: Files for submission should be .txt files with the class prediction for
    #  image M on line M. Note that image M corresponds to the Mth annotation in
    #  the provided annotation file. An example of a file in this format is
    #  train_perfect_preds.txt

    # save the image paths
    image_paths_file = os.path.join(args.model_dir, 'image_paths_file.csv')
    df = pd.DataFrame(data=image_paths)
    df.to_csv(image_paths_file, index=None, header=False)

    # save the predictions
    predictions_file = os.path.join(args.model_dir, 'predictions_file.csv')
    df = pd.DataFrame(data=predictions)
    df.to_csv(predictions_file, index=None, header=False)

    print('Done testing!')



