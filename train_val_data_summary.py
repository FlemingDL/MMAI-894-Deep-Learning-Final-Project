''' Create a report of which files were used for training and for validation'''
import os
import scipy.io as spio
import pandas as pd
import ssl
import slack
import logging


def post_slack_message(message, response=None):
    if 'SLACK_API_TOKEN' in os.environ:
        if response is None:
            try:
                response = client.chat_postMessage(channel=slack_channel, text=message)
            except:
                logging.info('Error posting message to slack')
        else:
            try:
                response = client.chat_update(channel=response['channel'], ts=response['ts'], text=message)
            except:
                logging.info('Error posting message to slack')

        return response


def post_slack_file(file_name, response=None):
    if 'SLACK_API_TOKEN' in os.environ:
        if response is None:
            try:
                response = client.files_upload(channels=slack_channel, file=file_name, filename=file_name)
            except:
                logging.info('Error uploading file to slack')
        else:
            delete_slack_file(response)
            try:
                response = client.files_upload(channels=slack_channel, file=file_name, filename=file_name)
            except:
                logging.info('Error uploading file to slack')

        return response


def delete_slack_message(response):
    if 'SLACK_API_TOKEN' in os.environ:
        try:
            client.chat_delete(channel=response['channel'], ts=response['ts'])
        except:
            logging.info('Error deleting message')


def delete_slack_file(response):
    if 'SLACK_API_TOKEN' in os.environ:
        client.files_delete(file=response['file']['id'])


if __name__ == '__main__':

    # Set slack channel
    slack_channel = '#dl-model-progress'
    # slack_channel = '#temp'

    if 'SLACK_API_TOKEN' in os.environ:
        # Setup slack messages to track progress
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        client = slack.WebClient(token=os.environ['SLACK_API_TOKEN'], ssl=ssl_context)

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
    post_slack_message(slack_message)
    post_slack_file(train_val_data_summary_file)

    print('Done Exporting')
