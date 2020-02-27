"""Visualize the outputs from intermediate layers for trained models
---
Creates visualizations of the first 25 filters for each layer in the network.  Images for each layer and
a text file for explaining the layers is saved in a visualizations folder and zipped

Example:
    python visualize_outputs_from_intermediate_layers.py --model_dir
        <directory of experiment with checkpoint file> --image_path <path of image file to run through model>

    if you have a slack api token (check the channel setting in code):
    SLACK_API_TOKEN='place token here' python visualize_outputs_from_intermediate_layers.py --model_dir
        <directory of experiment with checkpoint file> --image_path <path of image file to run through model>

---
"""
import argparse
import os
import shutil

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

import model_handler as mh
import utils
from slack_manager import SlackManager

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default=os.path.join('data', 'test', 'no_label', '00109.jpg'),
                    help="Path of image to create visual")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")


def input_number(message):
    user_input = 0
    while user_input < 1:
        try:
            user_input = int(input(message))
        except ValueError:
            print("Please enter an integer! Try again.")
    return user_input


class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = list(model.children())[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu().data.numpy()

    def remove(self):
        self.hook.remove()


def create_image_plot(activations, file_name):
    num_of_channels = activations.shape[1]
    num_of_plots = min(25, num_of_channels)
    fig = plt.figure(figsize=(7.5, 5))
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0.2, wspace=0.2)
    for i in range(num_of_plots):
        ax = fig.add_subplot(5, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(activations[0][i])
    image_file = os.path.join(args.model_dir, 'visualizations', file_name + ".png")
    fig.savefig(image_file)
    return image_file


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':

    # Initialize slack reporting
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

    # Set variables
    img_path = args.image_path
    model_name = params.model_name
    batch_size = params.test_batch_size
    num_workers = params.num_workers

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    print('Loading model...')
    model_ft = mh.load_checkpoint(filepath=args.model_dir, device=device)

    # Get the required input size of the network for resizing images
    input_size = mh.input_size_of_model(model_name)

    # Data augmentation and normalization image
    data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_pil = Image.open(img_path)
    image_tensor = data_transforms(img_pil).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor)
    input_img = input_img.to(device)

    layers = list(model_ft.children())

    vis_folder = os.path.join(args.model_dir, 'visualizations')
    ensure_folder(vis_folder)

    layer_summary = open(os.path.join(vis_folder, 'layer_summary.txt'), 'w')
    for i, layer in enumerate(layers):
        print('Working on layer {} of {}'.format(i, len(layers)-1))
        layer_summary.writelines('Layer {}\n'.format(i+1))
        layer_summary.writelines(str(layer))
        layer_summary.write('\n\n')
        if i+1 != len(layers):
            convolution_out = LayerActivations(model_ft, i)
            output = model_ft(Variable(input_img))
            convolution_out.remove()
            activations = convolution_out.features
            if activations is not None:
                create_image_plot(activations, 'visualization_of_layer_{}'.format(i+1))

    layer_summary.close()

    # zip files
    zip_file = shutil.make_archive(base_name=vis_folder, format='zip', root_dir=vis_folder)

    sm.post_slack_message('Here are the first 25 filters for each layer of experiment *{}* '
                          'using image {} in the train data set'
                          .format(args.model_dir, os.path.basename(img_path)))
    sm.post_slack_file(zip_file)

    print('Done.')
