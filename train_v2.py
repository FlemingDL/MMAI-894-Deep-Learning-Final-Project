"""Train the model"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
# import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os
import copy
import logging
import utils
import ssl
from tqdm import tqdm
import slack

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--slack_key', default='none',
                    help="Slack api key for messages")

"""
From https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

Model Training and Validation Code

The train_model function handles the training and validation of a given model. As input, it takes a PyTorch model,
a dictionary of dataloaders, a loss function, an optimizer, a specified number of epochs to train and validate for,
and a boolean flag for when the model is an Inception model. The is_inception flag is used to accomodate the
Inception v3 model, as that architecture uses an auxiliary output and the overall model loss respects both the
auxiliary output and the final output, as described here:
<https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958>__. The function
trains for the specified number of epochs and after each epoch runs a full validation step. It also keeps track of
the best performing model (in terms of validation accuracy), and at the end of training returns the best performing
model. After each epoch, the training and validation accuracies are printed.
"""


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

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
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
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
            post_slack_message('Epoch {}/{} {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, num_epochs - 1,
                                                                                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                val_acc_history.append(epoch_acc)

        print('')

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    post_slack_message('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    post_slack_message('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_acc_history


"""
From https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

Set Model Parameters’ .requires_grad attribute

This helper function sets the .requires_grad attribute of the parameters in the model to False when we are feature 
extracting. By default, when we load a pretrained model all of the parameters have .requires_grad=True, which is fine 
if we are training from scratch or finetuning. However, if we are feature extracting and only want to compute 
gradients for the newly initialized layer then we want all of the other parameters to not require gradients. This 
will make more sense later.
"""


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


"""
From https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

Initialize and Reshape the Networks

Now to the most interesting part. Here is where we handle the reshaping of each network. Note, this is not an 
automatic procedure and is unique to each model. Recall, the final layer of a CNN model, which is often times an FC 
layer, has the same number of nodes as the number of output classes in the dataset. Since all of the models have been 
pretrained on Imagenet, they all have output layers of size 1000, one node for each class. The goal here is to 
reshape the last layer to have the same number of inputs as before, AND to have the same number of outputs as the 
number of classes in the dataset. In the following sections we will discuss how to alter the architecture of each 
model individually. But first, there is one important detail regarding the difference between finetuning and 
feature-extraction. When feature extracting, we only want to update the parameters of the last layer, or in other 
words, we only want to update the parameters for the layer(s) we are reshaping. Therefore, we do not need to compute 
the gradients of the parameters that we are not changing, so for efficiency we set the .requires_grad attribute to 
False. This is important because by default, this attribute is set to True. Then, when we initialize the new layer 
and by default the new parameters have .requires_grad=True so only the new layer’s parameters will be updated. When 
we are finetuning we can leave all of the .required_grad’s set to the default of True. Finally, notice that 
inception_v3 requires the input size to be (299,299), whereas all of the other models expect (224,224).
"""


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


def post_slack_message(message):
    if 'SLACK_API_TOKEN' in os.environ:
        client.chat_postMessage(channel='CU3JRRA4E', text=message)


if __name__ == '__main__':

    # Ignore ssl certification (prevent error for some users)
    ssl._create_default_https_context = ssl._create_unverified_context

    # Collect arguments from command-line options
    args = parser.parse_args()

    if 'SLACK_API_TOKEN' in os.environ:
        print('Heyyyy')
        # Setup slack messages to track progress
        # client = slack.WebClient(token=os.environ['SLACK_API_TOKEN'])
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        client = slack.WebClient(token=os.environ['SLACK_API_TOKEN'], ssl=ssl_context)
        post_slack_message("New deep learning training started")

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    ######################################################################
    # Code block taken from From https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    # Variables set in experiment/base_model/params.json
    # ---------
    print('Setting variables...')
    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = "./data/"

    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "inception"
    post_slack_message('Model: {}'.format(model_name))

    # Number of classes in the dataset
    num_classes = 196

    # Batch size for training (change depending on how much memory you have)
    batch_size = 16

    # Number of epochs to train for
    num_epochs = 75

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False

    ######################################################################
    # Code block taken from From https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    # ---------

    print('Initializing the model...')

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # print the model we just instantiated
    print('Model loaded.')

    # #####################################################################
    # Code block taken from From https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    # ---------
    # Load Data
    # Now that we know what the input size must be, we can initialize the data transforms, image datasets,
    # and the dataloaders. Notice, the models were pretrained with the hard-coded normalization values, as described
    # here <https://pytorch.org/docs/master/torchvision/models.html>__.

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
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # #####################################################################
    # Code block taken from From https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    # ---------
    # Create the Optimizer
    # Now that the model structure is correct, the final step for finetuning and feature extracting is to
    # create an optimizer that only updates the desired parameters. Recall that after loading the pretrained model,
    # but before reshaping, if feature_extract=True we manually set all of the parameter’s .requires_grad attributes
    # to False. Then the reinitialized layer’s parameters have .requires_grad=True by default. So now we know that
    # all parameters that have .requires_grad=True should be optimized. Next, we make a list of such parameters and
    # input this list to the SGD algorithm constructor. To verify this, check out the printed parameters to learn.
    # When finetuning, this list should be long and include all of the model parameters. However, when feature
    # extracting this list should be short and only include the weights and biases of the reshaped layers.

    print('Create the optimizer...')

    # Send the model to GPU
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
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # #####################################################################
    # Code block taken from From https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    # ---------
    # Run Training and Validation Step
    # Finally, the last step is to setup the loss for the model, then run the training and validation function for
    # the set number of epochs. Notice, depending on the number of epochs this step may take a while on a CPU. Also,
    # the default learning rate is not optimal for all of the models, so to achieve maximum accuracy it would be
    # necessary to tune for each model separately.

    print('Running training and validation step...')

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=75,
                                 is_inception=(model_name == "inception"))

    print('Training complete.')
