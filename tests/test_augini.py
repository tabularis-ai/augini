import os
import pytest
import pandas as pd
import warnings
import augini as au
from augini.exceptions import APIError, DataProcessingError

# Filter out specific warnings
warnings.filterwarnings('ignore', category=ResourceWarning, message='unclosed.*<socket.socket.*>')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers.*')

# Test data fixture
@pytest.fixture
def test_df():
    """Create a test DataFrame for all tests"""
    return pd.DataFrame({
        'Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'Age': [30, 25, 45],
        'City': ['New York', 'Los Angeles', 'Chicago']
    })

@pytest.fixture
def api_key():
    """Get API key from environment"""
    key = os.environ.get('OPENROUTER_TOKEN')
    if not key:
        pytest.skip("No API key provided")
    return key

@pytest.fixture
def augmenter(api_key):
    """Create an Augment instance for testing"""
    return au.Augment(
        api_key=api_key,
        model='gpt-4o-mini',
        use_openrouter=True,
        debug=False
    )

@pytest.fixture
def chat(api_key, test_df):
    """Create a Chat instance for testing"""
    return au.Chat(
        df=test_df,
        api_key=api_key,
        model='gpt-4o-mini',
        use_openrouter=True,
        debug=False
    )

# Augment Tests
class TestAugment:
    """Test suite for the Augment class functionality"""

    def test_augment_single(self, augmenter, test_df, caplog):
        """Test single column augmentation with default prompt"""
        try:
            result_df = augmenter.augment_single(
                df=test_df,
                column_name='Occupation',
                custom_prompt="Based on the person's Name, Age, and City, suggest a realistic occupation. Return a JSON with the key 'Occupation'."
            )
            assert 'Occupation' in result_df.columns
            print(f"Generated 'Occupation' column: {result_df['Occupation'].tolist()}")
        except (APIError, DataProcessingError) as e:
            pytest.fail(f"augment_single raised {type(e).__name__} unexpectedly: {str(e)}")

    def test_augment_multiple(self, augmenter, test_df, caplog):
        """Test multiple column augmentation with default prompt"""
        try:
            result_df = augmenter.augment_columns(
                df=test_df,
                columns=['Hobby', 'FavoriteColor'],
                custom_prompt="Based on the person's profile, suggest a hobby and favorite color. Return a JSON with keys 'Hobby' and 'FavoriteColor'."
            )
            assert 'Hobby' in result_df.columns
            assert 'FavoriteColor' in result_df.columns
            print(f"Generated columns:")
            print(f"Hobbies: {result_df['Hobby'].tolist()}")
            print(f"Colors: {result_df['FavoriteColor'].tolist()}")
        except (APIError, DataProcessingError) as e:
            pytest.fail(f"augment_columns raised {type(e).__name__} unexpectedly: {str(e)}")

    def test_custom_prompt(self, augmenter, test_df, caplog):
        """Test single column augmentation with custom prompt"""
        custom_prompt = "Based on the person's name and age, suggest a quirky pet for them. Return a JSON with the key 'QuirkyPet'."
        try:
            result_df = augmenter.augment_single(
                df=test_df,
                column_name='QuirkyPet',
                custom_prompt=custom_prompt
            )
            assert 'QuirkyPet' in result_df.columns
            print(f"Generated quirky pets: {result_df['QuirkyPet'].tolist()}")
        except (APIError, DataProcessingError) as e:
            pytest.fail(f"augment_single with custom prompt raised {type(e).__name__} unexpectedly: {str(e)}")

    def test_invalid_api_key(self, test_df):
        """Test error handling with invalid API key"""
        invalid_augmenter = au.Augment(api_key="invalid_key", use_openrouter=True)
        with pytest.raises(APIError, match="Authentication failed"):
            invalid_augmenter.augment_single(
                df=test_df,
                column_name='InvalidFeature',
                custom_prompt="Generate an invalid feature. Return a JSON with the key 'InvalidFeature'."
            )

# Chat Tests
class TestChat:
    """Test suite for the Chat class functionality"""

    def test_basic_query(self, chat, caplog):
        """Test basic chat query functionality"""
        try:
            response = chat("What is the average age in the dataset?")
            assert isinstance(response, str)
            assert len(response) > 0
            print(f"Received response: {response[:100]}...")
        except APIError as e:
            pytest.fail(f"Basic query raised APIError unexpectedly: {str(e)}")

    def test_invalid_api_key(self, test_df):
        """Test error handling with invalid API key in chat"""
        invalid_chat = au.Chat(
            df=test_df,
            api_key="invalid_key",
            use_openrouter=True
        )
        with pytest.raises(APIError):
            invalid_chat("What is the average age?")

    def test_conversation_history_management(self, api_key, test_df, caplog):
        """Test conversation history management features"""
        chat_with_memory = au.Chat(
            df=test_df,
            api_key=api_key,
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
        assert len(full_history) == 2
        print(f"Full history count: {len(full_history)}")
        
        # Test summary history
        summary_history = chat_with_memory.get_conversation_history('summary')
        assert len(summary_history) == 2
        print(f"Summary history count: {len(summary_history)}")
        
        # Test clearing history
        chat_with_memory.clear_conversation_history('all')
        empty_history = chat_with_memory.get_conversation_history('full')
        assert len(empty_history) == 0
        print("Successfully cleared conversation history")