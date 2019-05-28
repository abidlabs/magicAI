import sys
import pandas as pd


def read_data(file_path, data_type):
    """
    reads data from file depending on type, to object that can be processed
    :param file_path: a string with the path to file that should be read
    :param data_type: a string with the type of data in file (one of: [`numerical`])
    :return: a DataFrame with the loaded data (pandas df, etc)
    """
    if data_type == 'numerical':
        try:
            data = pd.read_csv(file_path)
            return data
        except ValueError:
            print(f'ValueError: Could not read {file_path}. For numerical data use csv, json, or html files.')
    else:
        raise ValueError(f'{data_type} not a valid data_type.')


def summarize_data(loaded_data):
    """
    summarizes analysis of data (mean, median etc)
    :param loaded_data: a DataFrame which consists of the loaded data to be summarized
    :return: Pandas data description of DataFrame
    """
    return loaded_data.describe(include='all')


def compare_data(first_dataset_path, second_dataset_path, data_type, verbose=False):
    """
    main analysis function that creates the comparison of data summaries
    :param first_dataset_path: a string with the path to the first file to compare (ex: representing training data)
    :param second_dataset_path: a string with the path to the second file to compare (ex: representing deployment data)
    :param data_type: a string with the type of data in the files to compare (ex: numerical)
    :param verbose: a boolean that controls whether summary information should be printed.
    :return: a list with two items which are the data descriptions for the files
    """
    first_dataset_df = read_data(first_dataset_path, data_type)
    second_dataset_df = read_data(second_dataset_path, data_type)
    first_dataset_summary = summarize_data(first_dataset_df)
    second_dataset_summary = summarize_data(second_dataset_df)
    analysis = [first_dataset_summary, second_dataset_summary]

    if verbose:
        print(f'A summary of data in {first_dataset_path} (your first file)..')
        print(first_dataset_summary)
        print(f'A summary of data in {second_dataset_path} (your second file)..')
        print(second_dataset_summary)
    return analysis


if __name__ == "__main__":
    if len(sys.argv) == 5:
        compare_data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        compare_data(sys.argv[1], sys.argv[2], sys.argv[3])