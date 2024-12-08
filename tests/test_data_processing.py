import unittest

import pandas as pd

from src.scripts.process_data import transform_data


class TestDataProcessing(unittest.TestCase):
    def test_transform_data(self):
        """Test transformation of data based on skewness."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'B': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1000],
            'C': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            'Class': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        transformed_df = transform_data(df)
        self.assertEqual(transformed_df.shape, df.shape)
        self.assertTrue(all(transformed_df['Class'] == df['Class']))

    def test_transform_data_no_skew(self):
        """Test that data remains unchanged when no skewness is present."""
        df = pd.DataFrame({
            'A': [1, 1, 1, 1],
            'B': [1, 1, 1, 1],
            'C': [0, 0, 0, 0],
            'Class': [0, 1, 0, 1]
        })
        transformed_df = transform_data(df)
        pd.testing.assert_frame_equal(transformed_df, df, "Data should remain unchanged when no skewness")


if __name__ == '__main__':
    unittest.main()
