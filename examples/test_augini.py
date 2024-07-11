import unittest
import pandas as pd
from augini import Augini
from augini.exceptions import APIError, DataProcessingError

class TestAugini(unittest.TestCase):
    def setUp(self):
        self.api_key = "your_api_key"  # Replace with a valid API key for testing
        self.augini = Augini(api_key=self.api_key, use_openrouter=True, model='meta-llama/llama-3-8b-instruct', debug=False)
        self.df = pd.DataFrame({
            'Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'Age': [30, 25, 45],
            'City': ['New York', 'Los Angeles', 'Chicago']
        })

    def test_augment_single(self):
        try:
            result_df = self.augini.augment_columns(self.df, ['Occupation'])
            self.assertIn('Occupation', result_df.columns)
        except (APIError, DataProcessingError) as e:
            self.fail(f"augment_single raised {type(e).__name__} unexpectedly: {str(e)}")

    def test_augment_multiple(self):
        try:
            result_df = self.augini.augment_columns(self.df, ['Hobby', 'FavoriteColor'])
            self.assertIn('Hobby', result_df.columns)
            self.assertIn('FavoriteColor', result_df.columns)
        except (APIError, DataProcessingError) as e:
            self.fail(f"augment_columns raised {type(e).__name__} unexpectedly: {str(e)}")

    def test_custom_prompt(self):
        custom_prompt = "Based on the person's name and age, suggest a quirky pet for them. Respond with a JSON object with the key 'QuirkyPet'."
        try:
            result_df = self.augini.augment_columns(self.df, ['QuirkyPet'], custom_prompt=custom_prompt)
            self.assertIn('QuirkyPet', result_df.columns)
        except (APIError, DataProcessingError) as e:
            self.fail(f"augment_columns with custom prompt raised {type(e).__name__} unexpectedly: {str(e)}")

    def test_invalid_api_key(self):
        invalid_augini = Augini(api_key="invalid_key", use_openrouter=True)
        with self.assertRaises(APIError):
            invalid_augini.augment_columns(self.df, ['InvalidFeature'])

if __name__ == '__main__':
    unittest.main()