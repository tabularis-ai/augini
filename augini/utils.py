import json
import re

def extract_json(response):
    try:
        json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        return None

def generate_default_prompt(feature_names: list, available_columns: list) -> str:
    """
    Generate a default prompt for feature generation.
    
    Args:
        feature_names (list): List of features to generate
        available_columns (list): List of available columns in the DataFrame
        
    Returns:
        str: Generated prompt template
    """
    # Create a list of available data points
    data_points = ", ".join(f"{col}: {{{col}}}" for col in available_columns)
    
    # Create a list of features to generate
    features_list = ", ".join(feature_names)
    
    # Build the prompt template
    prompt = (
        f"Based on the following information:\n"
        f"{data_points}\n\n"
        f"Please generate realistic values for: {features_list}\n\n"
        f"Ensure the response is a valid JSON object with the following keys: {features_list}\n"
        f"Make the generated values realistic and consistent with the provided information.\n"
        f"If a value cannot be determined, use a reasonable default based on the available context."
    )
    
    return prompt


