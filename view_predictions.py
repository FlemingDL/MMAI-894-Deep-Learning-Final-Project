"""Creates a folder structure that shows the test image in the val folder, shows what it looks like transformed,
create a folder for each experiment provided and shows the prediction and transformed images use to train model.

In order to run, you need to have first ran predictions_and_accuracy.py for each model and train_val_data_summary.py

The code below has been hardwired for models in experiments/inception_model_sgd and experiments/resnet_model_sgd
---

Example:
    python view_predictions.py

---
"""

import os
import shutil
import tarfile

import cv2 as cv
import pandas as pd
import torchvision.utils
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def save_image(src_folder, file_name, dst_path):
    src_path = os.path.join(src_folder, file_name)
    img_pil = Image.open(src_path).convert('RGB')
    src_image = data_transforms(img_pil)
    ensure_folder(dst_path)
    dst_path = os.path.join(dst_path, file_name)
    torchvision.utils.save_image(src_image, dst_path)


def save_original_image(src_folder, file_name, dst_path):
    src_path = os.path.join(src_folder, file_name)
    src_image = cv.imread(src_path)
    ensure_folder(dst_path)
    dst_path = os.path.join(dst_path, "original.jpg")
    cv.imwrite(dst_path, src_image)


df_resnet = pd.read_csv('experiments/resnet_model_sgd/predictions.csv')
df_resnet['correct'] = df_resnet['label_class_id'] == df_resnet['predicted_class_id']
df_resnet = df_resnet.add_prefix('resnet_')
df_inception = pd.read_csv('experiments/inception_model_sgd/predictions.csv')
df_inception['correct'] = df_inception['label_class_id'] == df_inception['predicted_class_id']
df_inception = df_inception.add_prefix('inception_')

df_predicts = pd.concat([df_resnet, df_inception], axis=1, join='inner')
print(df_predicts.head())
# df_predicts = pd.read_csv('inception_resnet_sgd_prediction.csv')

# Extract the zip files if they haven't been extracted
print('Checking training set...')
if not os.path.exists('cars_train'):
    print('No cars_train folder.  Extracting cars_train.tgz')
    with tarfile.open('cars_train.tgz', "r:gz") as tar:
        tar.extractall()

for phase in ['both-incorrect', 'only-resnet-incorrect', 'only-inception-incorrect']:
    print(phase)

    if phase == 'both-incorrect':
        results = df_predicts[(df_predicts['inception_correct'] == False) & (df_predicts['resnet_correct'] == False)]
    elif phase == 'only-resnet-incorrect':
        results = df_predicts[(df_predicts['inception_correct'] == True) & (df_predicts['resnet_correct'] == False)]
    elif phase == 'only-inception-incorrect':
        results = df_predicts[(df_predicts['inception_correct'] == False) & (df_predicts['resnet_correct'] == True)]

    train_set = pd.read_csv('train_val_data_summary.csv')
    train_set = train_set[(train_set['phase'] == 'train')]

    file_source = 'cars_train'

    with tqdm(total=len(results.index)) as pbar:
        for i, row in results.iterrows():
            fname_len = len(str(row['resnet_file_name']))
            fname = "0" * (5 - fname_len) + str(row['resnet_file_name']) + '.jpg'
            save_original_image(file_source,
                                fname,
                                os.path.join('results', phase,
                                             'class-' + str(row['resnet_label_class_id']) + '-file-' + str(
                                                 row['resnet_file_name']))
                                )
            save_image(file_source,
                       fname,
                       os.path.join('results', phase,
                                    'class-' + str(row['resnet_label_class_id']) + '-file-' + str(
                                        row['resnet_file_name']))
                       )

            # do for inception
            incep_pred_class = int(row['inception_predicted_class_id'])
            train_set_for_class = train_set[(train_set['class_id'] == incep_pred_class)]
            for i, train_fname in train_set_for_class.iterrows():
                save_image(file_source,
                           train_fname['file_name'],
                           os.path.join('results', phase,
                                        'class-' + str(row['resnet_label_class_id']) + '-file-' + str(
                                            row['resnet_file_name']),
                                        'incep-predicted_class-' + str(train_fname['class_id']) + '-train_data'))

            # do for resnet
            resnet_pred_class = int(row['resnet_predicted_class_id'])
            train_set_for_class = train_set[(train_set['class_id'] == resnet_pred_class)]
            for i, train_fname in train_set_for_class.iterrows():
                save_image(file_source,
                           train_fname['file_name'],
                           os.path.join('results', phase,
                                        'class-' + str(row['resnet_label_class_id']) + '-file-' + str(
                                            row['resnet_file_name']),
                                        'resnet-predicted_class-' + str(train_fname['class_id']) + '-train_data'))
            pbar.update()

# cleanup
print('Cleaning up')
shutil.rmtree('cars_train')

print('Done!')
