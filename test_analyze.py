import unittest
from analyze import read_data
import pandas as pd
import os

dir_path = os.path.dirname(os.path.abspath(__file__))

test_train_filename = dir_path + '/test_data/train.csv'
test_deploy_filename = dir_path + '/test_data/deploy.csv'


class AnalyzeTestCase(unittest.TestCase):
    def test_read_data(self):
        self.assertIsInstance(read_data(test_train_filename), pd.DataFrame)
        self.assertIsInstance(read_data(test_deploy_filename), pd.DataFrame)


if __name__ == '__main__':
    unittest.main()