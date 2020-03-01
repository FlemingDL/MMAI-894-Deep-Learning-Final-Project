# Team Fleming MMAI 894 Deep Learning Final Project
This is team fleming's deep learning final.  Code is written for Python 3.

## Getting Started
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
$ python pre_process.py
```
To run for a specific experiment:
 ```bash
$ python pre_process.py --model_dir 'experiments/<directory name>'
```

## Training
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


## Testing For Submission
To create a submission file to the Stanford Cars 196 Submission Site, run:
```bash
$ python test.py --model_dir <new directory path>
```
A file called 'submit_to_stanford_prediction.txt' will be produced in your experiment directory.  This file
is formatted for submission. 

After submitting to Stanford Cars Evaluation, the returned .mat file can be parsed with:
```bash
$ python parse_results_file.py --file_path <path of .mat file>
```


## Helper Files
To create a report of the files that were used for training and validation:
```bash
$ python train_val_data_summary.py
```

To output a file of the prediction made for each image in the validation set and file showing
the accuracy for each class:
```bash
$ python predictions_and_accuracy.py --model_dir <directory of experiment>
```

To create a visuals of the first 25 filters in each of the layer:
```bash
$ python visualize_outputs_from_intermediate_layers.py
```

To view the images where the predictions for resnet/sgd and inception/sgd in experiments were incorrect, run:
```bash
$ python view_predictions.py
```
This will create a folder called 'results' and three sub-folders called 'both-incorrect' (both inception and resnet got
it wrong), 'only-inception-incorrect' (only inception got it wrong) and 'only-resnet-incorrect' (only resnet got it 
wrong). Within each of these subfolder will be series of folders with the following structure:
```bash
├── class-104-file-2199
│   ├── 02199.jpg
│   ├── incep-predicted_class-103-train_data
│   │   ├── 00262.jpg
│   │   ├── 00731.jpg
│   │   ├── ...
│   ├── original.jpg
│   └── resnet-predicted_class-103-train_data
│       ├── 00262.jpg
│       ├── 00731.jpg
│       ├── 00784.jpg
│       ├── ...
```
'class-104-file-2199' refers to class 104 and file name 02199.jpeg.  In this folder is the original image and the
transformed image.  The sub folder 'incep-predicted_class-103-train_data' refers to inception predicted class 103 and
within the folder are transformed images used for training.

In order to run, you need to have first ran predictions_and_accuracy.py for each model and train_val_data_summary.py

This code has been hardwired for models in experiments/inception_model_sgd and experiments/resnet_model_sgd

## Deactivate Virtual Environment
When you're done working on the project, deactivate the virtual environment with `deactivate`.
