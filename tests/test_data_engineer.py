import pytest
import pandas as pd
import numpy as np
from augini import DataEngineer, FeatureSpec
from unittest.mock import patch


@pytest.fixture
def sample_df():
    """Create a realistic sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        # Customer data
        "customer_id": range(1, n_samples + 1),
        "age": np.random.randint(18, 80, n_samples),
        "gender": np.random.choice(["M", "F"], n_samples),
        "city": np.random.choice(["New York", "London", "Tokyo"], n_samples),
        
        # Transaction data
        "purchase_amount": np.random.normal(100, 30, n_samples),
        "items_bought": np.random.randint(1, 10, n_samples),
        "is_first_purchase": np.random.choice([True, False], n_samples),
        "payment_method": np.random.choice(["Credit Card", "PayPal", "Cash"], n_samples),
        
        # Temporal data
        "days_since_last_purchase": np.random.randint(1, 365, n_samples),
        "total_purchases_ytd": np.random.randint(0, 50, n_samples),
        
        # Engagement metrics
        "website_visits": np.random.randint(1, 100, n_samples),
        "email_opens": np.random.randint(0, 20, n_samples),
        "support_tickets": np.random.randint(0, 5, n_samples)
    })


@pytest.fixture
def engineer():
    """Create a DataEngineer instance."""
    return DataEngineer(api_key="test-key")


def test_generate_feature(engineer, sample_df):
    """Test generating a single feature."""
    with patch('augini.data_engineer.DataEngineer.generate_feature', return_value=pd.DataFrame({
        "engagement_score": [75.5] * len(sample_df)
    })) as mock_generate:
        
        result = engineer.generate_feature(
            df=sample_df,
            name="engagement_score",
            description="Calculate customer engagement score (0-100)",
            output_type="float",
            source_columns=["purchase_amount", "items_bought", "is_first_purchase"],
            constraints={"min": 0, "max": 100}
        )
        
        assert "engagement_score" in result.columns
        assert not result["engagement_score"].isna().any()


def test_generate_multiple_features(engineer, sample_df):
    """Test generating multiple features at once."""
    with patch('augini.data_engineer.DataEngineer.generate_features', return_value=pd.DataFrame({
        "purchase_frequency_score": [8.5] * len(sample_df),
        "customer_status": ["Active"] * len(sample_df)
    })) as mock_generate:
        
        features = [
            {
                "name": "purchase_frequency_score",
                "description": "Score indicating purchase frequency (0-10)",
                "output_type": "float",
                "constraints": {"min": 0, "max": 10}
            },
            {
                "name": "customer_status",
                "description": "Customer status based on activity",
                "output_type": "category",
                "constraints": {"categories": ["New", "Active", "Dormant"]}
            }
        ]
        
        result = engineer.generate_features(df=sample_df, features=features)
        
        assert "purchase_frequency_score" in result.columns
        assert "customer_status" in result.columns
        assert not result["purchase_frequency_score"].isna().any()
        assert not result["customer_status"].isna().any()


@pytest.mark.parametrize(
    "invalid_input,error_type,error_match",
    [
        (pd.DataFrame(), ValueError, "must not be empty"),
        (None, ValueError, "must be a pandas DataFrame"),
        ("not a dataframe", ValueError, "must be a pandas DataFrame"),
    ],
)
def test_invalid_inputs(engineer, invalid_input, error_type, error_match):
    """Test error handling for invalid inputs."""
    with pytest.raises(error_type, match=error_match):
        engineer.generate_feature(
            df=invalid_input,
            name="test_feature",
            description="test description",
            output_type="float"
        )


@pytest.mark.parametrize(
    "output_type,name,description,expected_error",
    [
        ("invalid", "test", "test desc", "Unsupported output type"),
        ("float", "", "test desc", "name cannot be empty"),
        ("float", "test", "", "description cannot be empty"),
    ],
)
def test_invalid_parameters(engineer, sample_df, output_type, name, description, expected_error):
    """Test error handling for invalid parameters."""
    with pytest.raises(ValueError, match=expected_error):
        engineer.generate_feature(
            df=sample_df,
            name=name,
            description=description,
            output_type=output_type
        )
