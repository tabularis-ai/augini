import pandas as pd
from augini import Augini
from augini.exceptions import APIError, DataProcessingError

def test_augini():
    # Initialize Augini
    api_key = "your_api_key"
    augini = Augini(api_key=api_key, use_openrouter=True, model='meta-llama/llama-3-8b-instruct', debug=False)

    # Create a sample DataFrame
    data = {
        'Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'Age': [30, 25, 45],
        'City': ['New York', 'Los Angeles', 'Chicago']
    }
    df = pd.DataFrame(data)

    # Test 1: Add a single feature
    try:
        result_df = augini.augment_single(df, 'Occupation')
    except (APIError, DataProcessingError) as e:
        print(f"Test 1 failed: {str(e)}")

    # Test 2: Add multiple features
    try:
        result_df = augini.augment_columns(df, 'Hobby', 'FavoriteColor')
    except (APIError, DataProcessingError) as e:
        print(f"Test 2 failed: {str(e)}")

    # Test 3: Add a feature with a custom prompt
    try:
        custom_prompt = "Based on the person's name and age, suggest a quirky pet for them. Respond with a JSON object with the key 'QuirkyPet'."
        result_df = augini.augment_single(df, 'QuirkyPet', custom_prompt=custom_prompt)
    except (APIError, DataProcessingError) as e:
        print(f"Test 3 failed: {str(e)}")

    # Test 4: Test error handling with an invalid API key
    try:
        invalid_augini = Augini(api_key="invalid_key", use_openrouter=True)
        invalid_augini.augment_single(df, 'InvalidFeature')
    except APIError:
        print("Test 4 passed: APIError caught as expected")

if __name__ == "__main__":
    test_augini()
