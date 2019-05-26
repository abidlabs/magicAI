import sys
import pandas as pd


def read_data(filename, data_type):
    """
    reads data from file depending on type, to something that can be processed
    :param filename: path to file that should be read
    :param data_type: type of data in file (numerical, image, etc)
    :return: loaded data (pandas df, etc)
    """
    if data_type == 'numerical':
        try:
            data = pd.read_csv(filename)
            return data
        except ValueError:
            print('ValueError: Could not read {}. For numerical data use csv, json, or html files.'.format(filename))
    else:
        raise ValueError('{} not a valid data_type.'.format(data_type))


def summarize_data(data):
    """
    summarizes analysis of data (mean, median etc)
    :param data: loaded data to be summarized
    :return: data description/summary

    """
    return data.describe(include='all')


def compare_data(train, deploy, data_type, verbose=False):
    """
    main analysis function that creates the comparison of data summaries
    :param train: path to file that represents training data
    :param deploy: path to file that represents deployed data
    :param data_type: type of data in the train and deploy files (numerical, image, etc)
    :return: two item comparison list where items are data description summaries for the files
    """
    train_df = read_data(train, data_type)
    deploy_df = read_data(deploy, data_type)

    train_summary = summarize_data(train_df)
    deploy_summary = summarize_data(deploy_df)

    analysis = [train_summary, deploy_summary]

    if verbose:
        print('A summary of your training data (first file)..')
        print(train_summary)
        print('A summary of your deployed data (second file)..')
        print(deploy_summary)

    return analysis


if __name__ == "__main__":
    if len(sys.argv) == 5:
        compare_data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        compare_data(sys.argv[1], sys.argv[2], sys.argv[3])