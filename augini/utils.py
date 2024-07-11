import json
import re

def extract_json(response):
    try:
        json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        return None

def generate_default_prompt(feature_names, available_columns):
    column_list = ", ".join(f"{col}: {{{col}}}" for col in available_columns)
    features = ", ".join(f'"{feature}": "<{feature}>"' for feature in feature_names)
    return (f"Given the following data:\n{column_list}\n"
            f"Please provide the following features in a JSON object:\n{features}\n"
            "If a feature is not applicable or cannot be determined, use null in the JSON.\n"
            "Ensure the response is a valid JSON object as it is very important.")


