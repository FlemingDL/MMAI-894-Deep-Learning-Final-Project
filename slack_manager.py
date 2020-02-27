import logging
import ssl

import slack


class SlackManager:
    """Class that loads the slack api and makes it cleaner to post messages and files.

    Example:
    ```
    sm = SlackManager(channel='#dl-model-progress')
    if 'SLACK_API_TOKEN' in os.environ:
        sm.setup(slack_api_token=os.environ['SLACK_API_TOKEN'])

    sm.post_slack_message(message)
    sm.post_slack_file(file)
    ```
    """

    client = None
    slack_channel = ""

    def __init__(self, channel):
        self.slack_channel = channel

    def setup(self, slack_api_token):
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        self.client = slack.WebClient(token=slack_api_token, ssl=ssl_context)

    def post_slack_message(self, message, response=None):
        if self.client is not None:
            if response is None:
                try:
                    response = self.client.chat_postMessage(channel=self.slack_channel, text=message)
                except:
                    logging.info('Error posting message to slack')
            else:
                try:
                    response = self.client.chat_update(channel=response['channel'], ts=response['ts'], text=message)
                except:
                    logging.info('Error posting message to slack')

            return response

    def post_slack_file(self, file_name, response=None):
        if self.client is not None:
            if response is None:
                try:
                    response = self.client.files_upload(channels=self.slack_channel, file=file_name, filename=file_name)
                except:
                    logging.info('Error uploading file to slack')
            else:
                self.delete_slack_file(response)
                try:
                    response = self.client.files_upload(channels=self.slack_channel, file=file_name, filename=file_name)
                except:
                    logging.info('Error uploading file to slack')

            return response

    def delete_slack_message(self, response):
        if self.client is not None:
            try:
                self.client.chat_delete(channel=response['channel'], ts=response['ts'])
            except:
                logging.info('Error deleting message')

    def delete_slack_file(self, response):
        if self.client is not None:
            self.client.files_delete(file=response['file']['id'])
