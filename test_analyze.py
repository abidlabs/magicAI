import unittest
import analyze
import pandas as pd
import os

dir_path = os.path.dirname(os.path.abspath(__file__))

TRAIN_TESTFILE = 'test_data/train.csv'
DEPLOY_TESTFILE = 'test_data/deploy.csv'


class AnalyzeTestCase(unittest.TestCase):
    def test_read_data(self):
        self.assertIsInstance(analyze.read_data(TRAIN_TESTFILE, 'numerical'), pd.DataFrame)
        self.assertIsInstance(analyze.read_data(DEPLOY_TESTFILE, 'numerical'), pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
