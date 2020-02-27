"""Train and validate the model
---
Code leverages heavily from From https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

The code will train for the specified number of epochs and after each epoch runs a full validation step.
It also keeps track of the best performing model (in terms of validation accuracy), and at the end of training
saves the best performing model.

Variables for the model is set in the params.json in the experiment folder.

Available models are in model_handler.py
Code supports two optimizers, SGD and Adam (learning rate, momentum and eps are in the params.json file)

Example:
    python train.py --model_dir <directory of experiment>

    if you have a slack api token (check the channel setting in code):
    SLACK_API_TOKEN='place token here' python train.py --model_dir <directory of experiment>
---
"""
import argparse
import copy
import logging
import os
import shutil
import time

import markdown_strings as ms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from joke.jokes import *
from torchvision import datasets, transforms
from tqdm import tqdm

import model_handler as mh
import utils
from slack_manager import SlackManager

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Initialize slack image upload and message response
    slack_loss_img_response = None
    slack_acc_img_response = None
    slack_epoch_status_response = None
    epoch_time_elapsed = []

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        epoch_start = time.time()

        joke = ms.esc_format(icanhazdad())
        if epoch == 0:
            epoch_status_message = 'Just getting started. Chuck has a joke for you...' + joke
        else:
            ave_epoch_time = np.mean(epoch_time_elapsed)
            est_time_to_go = ave_epoch_time * (num_epochs - epoch) + ave_epoch_time
            epoch_status_message = '*Epoch {}/{}.*\n>' \
                                   'Est time to go {:.0f}m {:.0f}s\n>' \
                                   'Best accuracy: {:.2%}\n' \
                                   'Chuck has a new joke for you...{}'.format(epoch + 1,
                                                                              num_epochs,
                                                                              est_time_to_go // 60,
                                                                              est_time_to_go % 60,
                                                                              best_acc,
                                                                              joke)

        # update_slack_progress_bar(pbar, epoch, num_epochs)
        slack_epoch_status_response = sm.post_slack_message(epoch_status_message,
                                                            slack_epoch_status_response)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary
                        # -classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # save the training history
            if phase == 'train':
                train_acc_history.append(epoch_acc.item())
                train_loss_history.append(epoch_loss)
            if phase == 'val':
                val_acc_history.append(epoch_acc.item())
                val_loss_history.append(epoch_loss)

            # deep copy the the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # send graph every tenth epoch
        if epoch % 10 == 0 and epoch != 0:
            slack_loss_img_response = sm.post_slack_file(create_line_graph_image(train_loss_history, val_loss_history,
                                                                                 'train', 'validation',
                                                                                 'Loss Per Epoch'),
                                                         slack_loss_img_response)
            slack_acc_img_response = sm.post_slack_file(create_line_graph_image(train_acc_history, val_acc_history,
                                                                                'train', 'validation',
                                                                                'Accuracy Per Epoch'),
                                                        slack_acc_img_response)

        epoch_time_elapsed.append(time.time() - epoch_start)

        print('')

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    sm.post_slack_message('Training complete in {:.0f}m {:.0f}s\n'
                          'Best val Acc: {:4f}'.format(time_elapsed // 60, time_elapsed % 60, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_acc_history, val_loss_history, train_acc_history, train_loss_history


def create_line_graph_image(y1, y2, y1_name, y2_name, title):
    plt.figure(figsize=(10, 6))
    plt.plot(y1, label=y1_name)
    plt.plot(y2, label=y2_name)
    plt.legend()
    plt.title(title)
    image_file = os.path.join(args.model_dir, title.replace(" ", "_") + ".png")
    plt.savefig(image_file)
    return image_file


if __name__ == '__main__':

    # Setup slack
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
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    logging.info('Setting variables...')

    data_dir = "./data/"

    model_name = params.model_name
    optimizer_selected = params.optimizer
    learning_rate = params.learning_rate
    momentum = params.momentum
    eps = params.eps
    train_data_split = params.train_data_split
    images_cropped = params.crop_images_to_bounding_box

    # Number of classes in the dataset
    num_classes = 196

    # Batch size for training (change depending on how much memory you have)
    batch_size = params.batch_size

    # Number of epochs to train for
    num_epochs = params.num_epochs

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = params.feature_extract

    slack_message = "*New Training Started* {}\n>" \
                    "The parameters are...model: *{}*, optimizer: *{}*, learning rate: {}, " \
                    "momentum: {}, eps: {}, batch size: {}, number of epochs: {}, train and val data split: {}, " \
                    "images cropped to bounding box: {}\n" \
                    "We will need to call in Chuck Norris for this one...".format(args.model_dir,
                                                                                  model_name, optimizer_selected,
                                                                                  learning_rate, momentum, eps,
                                                                                  batch_size, num_epochs,
                                                                                  train_data_split, images_cropped)
    sm.post_slack_message(slack_message)

    # Initialize the model for this run
    logging.info('Initializing the model...')
    model_ft, input_size = mh.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    logging.info('Model loaded.')

    logging.info('Load data...')
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

    logging.info("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True,
                                       num_workers=params.num_workers) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info('Create the optimizer...')

    # Send the model to device (GPU or CPU)
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    if optimizer_selected == 'sgd':
        optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)

    elif optimizer_selected == 'adam':
        optimizer_ft = optim.Adam(params_to_update, lr=learning_rate, eps=eps)

    else:
        logging.info("Invalid optimizer name, exiting...")
        exit()

    logging.info('Running training and validation step...')

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, val_acc_history, val_loss_history, \
    train_acc_history, train_loss_history = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                                                        num_epochs=num_epochs,
                                                        is_inception=(model_name == "inception"))

    # Save the best model
    checkpoint = {
        'model': model_ft,
        'state_dict': model_ft.state_dict(),
        'optimizer': optimizer_ft.state_dict()
    }
    torch.save(checkpoint, os.path.join(args.model_dir, 'checkpoint.pt'))

    # save the validation accuracies
    val_accuracies_file = os.path.join(args.model_dir, 'validation_accuracies.csv')
    df = pd.DataFrame(data=val_acc_history)
    df.to_csv(val_accuracies_file, index=None, header=False)

    # save the validation losses
    val_loss_file = os.path.join(args.model_dir, 'validation_losses.csv')
    df = pd.DataFrame(data=val_loss_history)
    df.to_csv(val_loss_file, index=None, header=False)

    # save the train accuracies
    train_accuracies_file = os.path.join(args.model_dir, 'train_accuracies.csv')
    df = pd.DataFrame(data=train_acc_history)
    df.to_csv(train_accuracies_file, index=None, header=False)

    # save the train losses
    train_loss_file = os.path.join(args.model_dir, 'train_losses.csv')
    df = pd.DataFrame(data=train_loss_history)
    df.to_csv(train_loss_file, index=None, header=False)

    # zip files
    zip_file = shutil.make_archive(base_name=args.model_dir, format='zip', root_dir=args.model_dir)

    # post final validation accuracies to slack
    sm.post_slack_message('Here are the training results for the experiment:')
    sm.post_slack_file(zip_file)

    logging.info('Training complete.')
