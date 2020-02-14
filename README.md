# 894_DeepLearning_final_project_v2
This is team fleming's deep learning final project version 2

## Getting Started
Clone this repository
```bash
$ git clone https://github.com/FlemingDL/894_DeepLearning_final_project_v2.git
```
Change into the 894_DeepLearning_final_project
```bash
$ cd 894_DeepLearning_final_project_v2
```
Create a virtual environment.  If virtualenv is not installed on your system run
```bash
$ pip install virtualenv
```
When virtualenv is installed, create a virtual environment (I like to call it venv)
```bash
$ virtualenv venv
```
Activate the virtual environment
```bash
$ source venv/bin/activate
```
Install the packages in requirements.txt for this project
```bash
$ pip3 install -r requirements.txt
```

## Dataset
Download training data, test data, class labels and bounding boxes (for both train and test).
```bash
$ python download_data.py
```

### Data Pre-processing
Extract training images and split them by 80:20.  Images are cropped by the bounding boxes, resized to 224 x 224, and
saved into folders that are grouped by label.  The reason images of the same class are stored in folders of the same
class name is becuase the torchvision package includes a class called ImageFolder.  This class handles a lot of the work
providing the images are in a structure where each directory is a label.
```bash
$ python preprocess.py
```

### Train
The default experiment is in './experiments/base_model'.  In this folder is a set of training parameters in
'params.json'.  To run this experiment, run: 
```bash
$ python train.py
```
To create new experiment, create a new directory in './experiments/', create new parameters saved in a 'params.json'
file in your new directory.  Then run:
```bash
$ python train.py --model_dir <new directory path>
```

When you're done working on the project, deactivate the virtual environment with `deactivate`.
