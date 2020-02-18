"""
Hyperparameter tuning using Ax.

Code heavily taken from https://ax.dev/tutorials/tune_cnn.html

"""
import torch
import numpy as np
import argparse
import torch.nn as nn
from torchvision import datasets, models, transforms
import ssl
import logging
import os
import slack
import json

import utils

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.plot.render import plot_config_to_html
from ax.utils.report.render import render_report_elements
from ax.utils.tutorials.cnn_utils import train, evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        logging.info("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


"""
Code block taken from https://ax.dev/tutorials/tune_cnn.html

Define function to optimize

In this tutorial, we want to optimize classification accuracy on the validation set as a function of the learning 
rate and momentum. The function takes in a parameterization (set of parameter values), computes the classification 
accuracy, and returns a dictionary of metric name ('accuracy') to a tuple with the mean and standard error.
"""


def train_evaluate(parameterization):
    net = model
    net = train(net=net, train_loader=dataloaders_dict['train'], parameters=parameterization, dtype=dtype,
                device=device)
    return evaluate(
        net=net,
        data_loader=dataloaders_dict['val'],
        dtype=dtype,
        device=device,
    )


def post_slack_message(message):
    if 'SLACK_API_TOKEN' in os.environ:
        client.chat_postMessage(channel='#dl-model-progress', text=message)


def post_slack_file(file_name):
    if 'SLACK_API_TOKEN' in os.environ:
        client.files_upload(channels='#dl-model-progress', file=file_name, filename=file_name)


if __name__ == '__main__':
    # Ignore ssl certification (prevent error for some users)
    ssl._create_default_https_context = ssl._create_unverified_context

    # Collect arguments from command-line options
    args = parser.parse_args()

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    if 'SLACK_API_TOKEN' in os.environ:
        # Setup slack messages to track progress
        # client = slack.WebClient(token=os.environ['SLACK_API_TOKEN'])
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        client = slack.WebClient(token=os.environ['SLACK_API_TOKEN'], ssl=ssl_context)
        with open(json_path) as f:
            params_text = json.load(f)
        post_slack_message("Hyperparameter tuning started for experiment: {}".format(args.model_dir))

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'hyperparameter_tuning.log'))

    torch.manual_seed(12345)
    dtype = torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Setting variables...')
    data_dir = "./data/"
    model_name = params.model_name
    num_classes = 196
    batch_size = params.batch_size
    num_epochs = params.num_epochs
    feature_extract = params.feature_extract

    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    print('Load data...')
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True,
                                       num_workers=params.num_workers) for x in ['train', 'val']}

    # #####################################################################
    # Code block taken from https://ax.dev/tutorials/tune_cnn.html
    # ---------
    # Run the optimization loop
    # Here, we set the bounds on the learning rate and momentum and set the parameter space for the learning rate to
    # be on a log scale.

    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
            {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
        ],
        evaluation_function=train_evaluate,
        objective_name='accuracy'
    )

    optimal_lr = best_parameters['lr']
    optimal_momentum = best_parameters['momentum']

    logging.info('Optimal learning rate is: {}'.format(optimal_lr))
    logging.info('Optimal momentum is: {}'.format(optimal_momentum))

    means, covariances = values

    logging.info('Means: {}'.format(means))
    logging.info('Covariances: {}'.format(covariances))

    # #####################################################################
    # Code block taken from https://ax.dev/tutorials/tune_cnn.html
    # ---------
    # Plot response surface
    #
    # Contour plot showing classification accuracy as a function of the two hyperparameters.
    #
    # The black squares show points that we have actually run, notice how they are clustered in the optimal region.

    plot_config = plot_contour(model=model, param_x='lr', param_y='momentum', metric_name='accuracy')

    # create an Ax report
    with open(os.path.join(args.model_dir, 'plot_response_surface_image.html'), 'w') as outfile:
        outfile.write(render_report_elements(
            "Response Surface",
            html_elements=[plot_config_to_html(plot_config)],
            header=False,
        ))

    # #####################################################################
    # Code block taken from https://ax.dev/tutorials/tune_cnn.html
    # ---------
    # Plot best objective as function of the iteration¶
    #
    # Show the model accuracy improving as we identify better hyperparameters.

    # `plot_single_method` expects a 2-d array of means, because it expects to average means from multiple
    # optimization runs, so we wrap out best objectives array in another array.
    best_objectives = np.array([[trial.objective_mean * 100 for trial in experiment.trials.values()]])
    best_objective_plot = optimization_trace_single_method(
        y=np.maximum.accumulate(best_objectives, axis=1),
        title="Model performance vs. # of iterations",
        ylabel="Classification Accuracy, %",
    )

    # create an Ax report
    with open(os.path.join(args.model_dir, 'best_objective_plot.html'), 'w') as outfile:
        outfile.write(render_report_elements(
            "Best Objective",
            html_elements=[plot_config_to_html(best_objective_plot)],
            header=False,
        ))

    # save optimal parameters to params.json
    params.learning_rate = optimal_lr
    params.momentum = optimal_momentum
    params.save(json_path)

    # save html files as images
    post_slack_message('Optimization completed\nBest learning rate: {}\n'
                       'Best momentum: {}'.format(optimal_lr, optimal_momentum))
    post_slack_file(os.path.join(args.model_dir, 'plot_response_surface_image.html'))
    post_slack_file(os.path.join(args.model_dir, 'best_objective_plot.html'))
