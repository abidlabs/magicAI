import sys
import os
import pandas as pd


def read_data(filename):
    """
    loads data from file depending on type
    """
    file_ext = os.path.splitext(filename)[1]
    if file_ext == '.csv':
        data = pd.read_csv(filename)
    else:
        data = []
    return data


def summarize_data(data):
    """
    returns summary analysis of data (mean, median etc)
    """
    return data.describe(include='all')


def compare_data(train, deploy):
    """
    returns side by side comparison of data summaries
    """
    train_df = read_data(train)
    deploy_df = read_data(deploy)

    train_summary = summarize_data(train_df)
    deploy_summary = summarize_data(deploy_df)

    print('A summary of data from your first file..')
    print(train_summary)
    print('A summary of data from your first file..')
    print(deploy_summary)


if __name__ == "__main__":
    compare_data(sys.argv[1], sys.argv[2])

