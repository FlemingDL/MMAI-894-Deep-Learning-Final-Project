"""Test the model"""
import argparse
import ssl
import torch
import slack
import os
import utils
import json
from torchvision import transforms

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

    else:
        logging.info("Invalid model name, exiting...")
        exit()

    return input_size


def load_checkpoint(filepath, inference=False):
    checkpoint = torch.load(filepath + 'checkpoint.pt')
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

    data_dir = "./data/"
    model_name = params.model_name

    input_size = input_size_of_model(model_name)

    model_ft = load_checkpoint(filepath=args.model_dir)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = {
        'testing': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }



    model_ft = model_ft.to(device)
