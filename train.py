"""Train the model"""

import argparse
import os
import ssl
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from torchvision import datasets, models, transforms
from tqdm import tqdm

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
# parser.add_argument('--restore_file', default=None,
#                     help="Optional, name of the file in --model_dir containing weights "
#                          "to reload before training")  # 'best' or 'train'


def train_model(model, loss_fn, optimizer, scheduler):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(params.num_epochs):
        log_text = "Epoch {}/{}".format(epoch + 1, params.num_epochs)
        logging.info(log_text)
        print('-' * 10)

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

                # clear previous gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # compute model output
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # calculate loss
                    loss = loss_fn(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # compute gradients of all variables wrt loss
                        loss.backward()
                        # perform updates using calculated gradients
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            is_best = epoch_acc >= best_acc

            log_text = '{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)
            logging.info(log_text)

            checkpoint = {'epoch': epoch + 1,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()}

            # Save weights
            utils.save_checkpoint(checkpoint, is_best=is_best, checkpoint=args.model_dir)

            # deep copy the model
            if phase == 'val' and is_best:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    log_text = 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    logging.info(log_text)
    log_text = 'Best val Acc: {:4f}'.format(best_acc)
    logging.info(log_text)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':

    # Collect arguments from command-line options
    args = parser.parse_args()

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    ######################################################################
    # Set GPU or CPU training
    # ---------
    # If a GPU is available set the device to cuda
    params.cuda = torch.cuda.is_available()
    if params.cuda:
        device = torch.device('cuda')
        print('CUDA is available. Training on GPU: {}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device('cpu')
        print('CUDA is not available. Training on CPU ...')

    ######################################################################
    # Load Data
    # ---------
    #
    logging.info("Loading the datasets...")

    # Data augmentation and normalization for training
    # Just normalization for validation
    # Create the input data pipeline
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = './data/'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=params.batch_size,
                                                  shuffle=True,
                                                  num_workers=params.num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x])
                     for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    logging.info("- done.")

    ######################################################################
    # Finetuning the convolutional network
    # ---------
    # Load a pretrained model and reset final fully connected layer.
    #
    # Ignore ssl certification (prevent error for some users)
    ssl._create_default_https_context = ssl._create_unverified_context

    # Set the transfer model
    model = models.resnet18(pretrained=True)

    # If user selected to freeze the network and only train the final layer
    # then freeze the parameters so gradients are not computed in 'backward'
    if params.freeze_network:
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    if params.freeze_network:
        # Only parameters of final layer are being optimized
        optimizer = optim.SGD(model.fc.parameters(), lr=params.learning_rate, momentum=params.momentum)
    else:
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=params.momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    ######################################################################
    # Train and evaluate
    # ---------
    #
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_model(model, loss_fn, optimizer, exp_lr_scheduler)
