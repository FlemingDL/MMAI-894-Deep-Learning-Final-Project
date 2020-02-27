"""Script to parses the .mat file returned from the Cars 196 Submission site

Example:
    python parse_results_file.py --file_path <path of .mat file>

"""
import argparse

import pandas as pd
import scipy.io as spio

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', help="The path of the response file")

if __name__ == '__main__':

    # Collect arguments from command-line options
    args = parser.parse_args()

    if args.file_path is None:
        print('A file to parse must be provided.  Enter \'python parse_results_file.py --file_path <Path to file>\'')
        exit()

    file_data = spio.loadmat(args.file_path)

    print('Exporting confusion matrix to confusion_matrix.csv')
    cm = file_data['confusion_matrix']
    df = pd.DataFrame(cm)
    df.to_csv('confusion_matrix.csv', index=False, header=False)

    print('The reported accuracy was: {}'.format(file_data['accuracy'][0][0]))
