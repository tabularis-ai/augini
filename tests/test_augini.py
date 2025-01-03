import os
import unittest
import pandas as pd
import warnings
import asyncio
import augini as au
from augini.exceptions import APIError, DataProcessingError

# Filter out specific warnings
warnings.filterwarnings('ignore', category=ResourceWarning, message='unclosed.*<socket.socket.*>')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers.*')

class VerboseTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"\nRunning {cls.__name__} tests...")
        super().setUpClass()
    
    def setUp(self):
        print(f"\n{self._testMethodName}: {self._testMethodDoc or 'No description'}")
        super().setUp()

    def tearDown(self):
        # Clean up any remaining event loops
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.close()
        except Exception:
            pass
        super().tearDown()

class TestAugment(VerboseTestCase):
    """Test suite for the Augment class functionality"""
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.api_key = os.environ.get('OPENROUTER_TOKEN')
        if not cls.api_key:
            raise unittest.SkipTest("No API key provided")
        print("\nInitializing Augment test suite with test data...")
        cls.augmenter = au.Augment(
            api_key=cls.api_key,
            model='gpt-4o-mini',
            use_openrouter=True,
            debug=False
        )
        cls.df = pd.DataFrame({
            'Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'Age': [30, 25, 45],
            'City': ['New York', 'Los Angeles', 'Chicago']
        })

    def test_augment_single(self):
        """Test single column augmentation with default prompt"""
        try:
            result_df = self.augmenter.augment_single(
                df=self.df,
                column_name='Occupation',
                custom_prompt="Based on the person's Name, Age, and City, suggest a realistic occupation. Return a JSON with the key 'Occupation'."
            )
            self.assertIn('Occupation', result_df.columns)
            print(f"Successfully generated 'Occupation' column: {result_df['Occupation'].tolist()}")
        except (APIError, DataProcessingError) as e:
            self.fail(f"augment_single raised {type(e).__name__} unexpectedly: {str(e)}")

    def test_augment_multiple(self):
        """Test multiple column augmentation with default prompt"""
        try:
            result_df = self.augmenter.augment_columns(
                df=self.df,
                columns=['Hobby', 'FavoriteColor'],
                custom_prompt="Based on the person's profile, suggest a hobby and favorite color. Return a JSON with keys 'Hobby' and 'FavoriteColor'."
            )
            self.assertIn('Hobby', result_df.columns)
            self.assertIn('FavoriteColor', result_df.columns)
            print(f"Successfully generated columns:")
            print(f"Hobbies: {result_df['Hobby'].tolist()}")
            print(f"Colors: {result_df['FavoriteColor'].tolist()}")
        except (APIError, DataProcessingError) as e:
            self.fail(f"augment_columns raised {type(e).__name__} unexpectedly: {str(e)}")

    def test_custom_prompt(self):
        """Test single column augmentation with custom prompt"""
        custom_prompt = "Based on the person's name and age, suggest a quirky pet for them. Return a JSON with the key 'QuirkyPet'."
        try:
            result_df = self.augmenter.augment_single(
                df=self.df,
                column_name='QuirkyPet',
                custom_prompt=custom_prompt
            )
            self.assertIn('QuirkyPet', result_df.columns)
            print(f"Successfully generated quirky pets: {result_df['QuirkyPet'].tolist()}")
        except (APIError, DataProcessingError) as e:
            self.fail(f"augment_single with custom prompt raised {type(e).__name__} unexpectedly: {str(e)}")

    def test_invalid_api_key(self):
        """Test error handling with invalid API key"""
        invalid_augmenter = au.Augment(api_key="invalid_key", use_openrouter=True)
        with self.assertRaises(APIError) as context:
            invalid_augmenter.augment_single(
                df=self.df,
                column_name='InvalidFeature',
                custom_prompt="Generate an invalid feature. Return a JSON with the key 'InvalidFeature'."
            )
        self.assertIn("Authentication failed", str(context.exception))
        print("Successfully caught invalid API key error")


class TestChat(VerboseTestCase):
    """Test suite for the Chat class functionality"""
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.api_key = os.environ.get('OPENROUTER_TOKEN')
        if not cls.api_key:
            raise unittest.SkipTest("No API key provided")
        print("\nInitializing Chat test suite with test data...")
        cls.df = pd.DataFrame({
            'Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'Age': [30, 25, 45],
            'City': ['New York', 'Los Angeles', 'Chicago']
        })
        cls.chat = au.Chat(
            df=cls.df,
            api_key=cls.api_key,
            model='gpt-4o-mini',
            use_openrouter=True,
            debug=False
        )

    def test_basic_query(self):
        """Test basic chat query functionality"""
        try:
            response = self.chat("What is the average age in the dataset?")
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            print(f"Successfully received response: {response[:100]}...")
        except APIError as e:
            self.fail(f"Basic query raised APIError unexpectedly: {str(e)}")

    def test_invalid_api_key(self):
        """Test error handling with invalid API key in chat"""
        invalid_chat = au.Chat(
            df=self.df,
            api_key="invalid_key",
            use_openrouter=True
        )
        with self.assertRaises(APIError):
            invalid_chat("What is the average age?")
        print("Successfully caught invalid API key error")

    def test_conversation_history_management(self):
        """Test conversation history management features"""
        chat_with_memory = au.Chat(
            df=self.df,
            api_key=self.api_key,
            enable_memory=True
        )
        
        print("\nTesting conversation history...")
        # Make some queries
        response1 = chat_with_memory("What is the average age?")
        print(f"Query 1 response: {response1[:100]}...")
        response2 = chat_with_memory("List all unique cities.")
        print(f"Query 2 response: {response2[:100]}...")
        
        # Test full history
        full_history = chat_with_memory.get_conversation_history('full')
        self.assertEqual(len(full_history), 2)
        print(f"Full history count: {len(full_history)}")
        
        # Test summary history
        summary_history = chat_with_memory.get_conversation_history('summary')
        self.assertEqual(len(summary_history), 2)
        print(f"Summary history count: {len(summary_history)}")
        
        # Test clearing history
        chat_with_memory.clear_conversation_history('all')
        empty_history = chat_with_memory.get_conversation_history('full')
        self.assertEqual(len(empty_history), 0)
        print("Successfully cleared conversation history")


if __name__ == '__main__':
    # Run tests with more verbose output
    unittest.main(verbosity=2)