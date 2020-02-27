""" Creates a report of which files were used for training and for validation
---
Example:
    python train_val_data_summary.py

    if you have a slack api token (check the channel setting in code):
    SLACK_API_TOKEN='place token here' python train_val_data_summary.py
---
"""
import os

import pandas as pd
import scipy.io as spio

from slack_manager import SlackManager

if __name__ == '__main__':

    # Setup slack
    # sm = SlackManager(channel='#temp')
    sm = SlackManager(channel='#dl-model-progress')
    if 'SLACK_API_TOKEN' in os.environ:
        sm.setup(slack_api_token=os.environ['SLACK_API_TOKEN'])

    # Get the car name lookup table
    devkit = 'devkit'
    cars_meta_data = spio.loadmat(os.path.join(devkit, 'cars_meta.mat'))
    cars_classid_to_name = [c for c in cars_meta_data['class_names'][0]]
    cars_classid_to_name = pd.DataFrame(cars_classid_to_name, columns=['name'])

    phases = []
    class_ids = []
    class_names = []
    file_names = []

    df = pd.DataFrame()

    for phase in ['train', 'val']:
        print(phase)
        for root, dirs, files in os.walk(os.path.join('data', phase)):
            for name in files:
                phases.append(phase)
                class_id = int(os.path.basename(root))
                class_ids.append(class_id)
                class_names.append(cars_classid_to_name.iloc[class_id - 1]['name'])
                file_names.append(name)
                # print(os.path.join(root, name))

    df['phase'] = phases
    df['class_id'] = class_ids
    df['class_name'] = class_names
    df['file_name'] = file_names

    train_val_data_summary_file = 'train_val_data_summary.csv'
    df.to_csv(train_val_data_summary_file, index=None)

    slack_message = "*Data Summary Report of Train and Validation Data Used to Create All Models*"
    sm.post_slack_message(slack_message)
    sm.post_slack_file(train_val_data_summary_file)

    print('Done Exporting')
