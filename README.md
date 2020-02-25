# Team Fleming MMAI 894 Deep Learning Final Project
This is team fleming's deep learning final.  Code is written for Python 3.

## Getting Started
Clone this repository
```bash
$ git clone https://github.com/FlemingDL/MMAI-894-Deep-Learning-Final-Project.git
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
$ pip install -r requirements.txt
```

## Dataset
Download training data, test data, class labels and bounding boxes (for both train and test).
```bash
$ python download_data.py
```

### Data Pre-processing
Extract training images and split them into train and validate according to the ratio set in the params.json file in
the experiments sub-folder.  If the parameter 'crop_images_to_bounding_box' in params.json is set to 'true', images 
are first cropped by the bounding boxes and then saved into folders that are grouped by label.  The reason images of 
the same class are stored in folders of the same class name is because the torchvision package includes a class 
called ImageFolder.  This class handles a lot of the work providing the images are in a structure where each directory 
is a label.
```bash
$ python preprocess.py
```
To run for a specific experiment:
 ```bash
$ python preprocess.py --model_dir 'experiments/<directory name>'
```

### Train
The default experiment is in './experiments/base_model'.  In this folder is a set of training parameters in
'params.json'.  To run this experiment, run: 
```bash
$ python train.py
```
To create a new experiment, create a new directory in './experiments/', copy the 'params.json' file from 
'./experiments/base_model' to your new directory.  Edit the parameters in the params.json file in your new 
directory.  Then run:
```bash
$ python train.py --model_dir <new directory path>
```


### Test
To create a submission file to the Stanford Cars 196 Submission Site, run:
```bash
$ python test.py --model_dir <new directory path>
```
A file called 'submit_to_stanford_prediction.txt' will be produced in your experiment directory.  This file
is formatted for submission. 

When you're done working on the project, deactivate the virtual environment with `deactivate`.
