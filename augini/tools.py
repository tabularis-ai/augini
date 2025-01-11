from typing import List, Dict, Any, Optional, Callable
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class DataAnalysisTools:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def convert_to_json_serializable(self, obj):
        """Convert non-serializable objects to JSON-compatible types."""
        if isinstance(obj, (np.bool_, bool)):  # Use np.bool_ or bool
            return bool(obj)
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):  # Fix: Use np.ndarray only
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, (list, tuple)):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        else:
            return obj
        
    def get_column_stats(self, column_name: str) -> Dict[str, Any]:
        """Calculate comprehensive statistics for a specific column"""
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")
            
        col_data = self.df[column_name]
        stats_dict = {
            "type": str(col_data.dtype),
            "missing_count": col_data.isna().sum(),
            "missing_percentage": (col_data.isna().sum() / len(col_data)) * 100
        }
        
        if np.issubdtype(col_data.dtype, np.number):
            desc = col_data.describe()
            stats_dict.update({
                "mean": desc["mean"],
                "median": col_data.median(),
                "std": desc["std"],
                "min": desc["min"],
                "max": desc["max"],
                "quartiles": {
                    "25%": desc["25%"],
                    "50%": desc["50%"],
                    "75%": desc["75%"]
                },
                "skewness": col_data.skew(),
                "kurtosis": col_data.kurtosis()
            })
        else:
            value_counts = col_data.value_counts()
            stats_dict.update({
                "unique_values": len(value_counts),
                "most_common": value_counts.head(5).to_dict(),
                "least_common": value_counts.tail(5).to_dict()
            })
            
        return self.convert_to_json_serializable(stats_dict)

    def correlation_analysis(self, 
                           columns: Optional[List[str]] = None, 
                           method: str = 'pearson') -> Dict[str, Any]:
        """Analyze correlations between numeric columns"""
        if columns:
            df_subset = self.df[columns]
        else:
            df_subset = self.df.select_dtypes(include=[np.number])
            
        corr_matrix = df_subset.corr(method=method)
        
        # Get strongest correlations
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                correlations.append({
                    "column1": col1,
                    "column2": col2,
                    "correlation": corr_value,
                    "strength": "strong" if abs(corr_value) > 0.7 else 
                               "moderate" if abs(corr_value) > 0.4 else "weak"
                })
                
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        result = {
            "method": method,
            "correlations": correlations,
            "matrix": corr_matrix.to_dict()
        }

        return self.convert_to_json_serializable(result)

    def distribution_analysis(self, column_name: str) -> Dict[str, Any]:
        """Analyze the distribution of a column"""
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")
            
        data = self.df[column_name]
        
        if np.issubdtype(data.dtype, np.number):
            # Test for normality
            _, normality_p_value = stats.normaltest(data.dropna())
            
            # Calculate distribution metrics
            analysis = {
                "type": "numerical",
                "normality_test": {
                    "is_normal": normality_p_value > 0.05,
                    "p_value": normality_p_value
                },
                "distribution_metrics": {
                    "skewness": data.skew(),
                    "kurtosis": data.kurtosis(),
                    "variance": data.var(),
                    "std_dev": data.std()
                },
                "percentiles": {
                    str(i): np.percentile(data.dropna(), i)
                    for i in range(0, 101, 10)
                }
            }
        else:
            # For categorical data
            value_counts = data.value_counts()
            entropy = stats.entropy(value_counts.values)
            
            analysis = {
                "type": "categorical",
                "distribution_metrics": {
                    "entropy": entropy,
                    "unique_count": len(value_counts),
                    "mode": data.mode().iloc[0] if not data.mode().empty else None
                },
                "category_frequencies": value_counts.to_dict(),
                "category_percentages": (value_counts / len(data) * 100).to_dict()
            }
            
        return self.convert_to_json_serializable(analysis)

    def outlier_detection(self, 
                         column_name: str, 
                         method: str = 'zscore',
                         threshold: float = 3.0) -> Dict[str, Any]:
        """Detect outliers in a column using various methods"""
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")
            
        if not np.issubdtype(self.df[column_name].dtype, np.number):
            raise ValueError(f"Column '{column_name}' must be numeric")
            
        data = self.df[column_name].dropna()
        
        outliers = {}
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers_mask = z_scores > threshold
            outliers = {
                "method": "z-score",
                "threshold": threshold,
                "outlier_indices": outliers_mask.nonzero()[0].tolist(),
                "outlier_values": data[outliers_mask].tolist(),
                "outlier_count": sum(outliers_mask),
                "outlier_percentage": (sum(outliers_mask) / len(data)) * 100
            }
        elif method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers_mask = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
            outliers = {
                "method": "IQR",
                "Q1": Q1,
                "Q3": Q3,
                "IQR": IQR,
                "outlier_indices": outliers_mask.nonzero()[0].tolist(),
                "outlier_values": data[outliers_mask].tolist(),
                "outlier_count": sum(outliers_mask),
                "outlier_percentage": (sum(outliers_mask) / len(data)) * 100
            }
            
        return self.convert_to_json_serializable(outliers)

    def time_series_analysis(self, 
                           date_column: str, 
                           value_column: str,
                           freq: str = 'D') -> Dict[str, Any]:
        """Analyze time series data"""
        if date_column not in self.df.columns or value_column not in self.df.columns:
            raise ValueError("Specified columns not found in DataFrame")
            
        # Ensure date column is datetime
        df_ts = self.df.copy()
        df_ts[date_column] = pd.to_datetime(df_ts[date_column])
        
        # Sort by date
        df_ts = df_ts.sort_values(date_column)
        
        # Resample to regular frequency
        ts = df_ts.set_index(date_column)[value_column].resample(freq).mean()
        
        # Calculate basic time series metrics
        analysis = {
            "timeframe": {
                "start": ts.index.min().isoformat(),
                "end": ts.index.max().isoformat(),
                "duration_days": (ts.index.max() - ts.index.min()).days
            },
            "trends": {
                "overall_trend": "increasing" if ts.iloc[-1] > ts.iloc[0] else "decreasing",
                "mean": ts.mean(),
                "std": ts.std(),
                "min": ts.min(),
                "max": ts.max()
            }
        }
        
        # Add seasonality analysis if enough data points
        if len(ts) >= 4:
            # Simple seasonality check using autocorrelation
            acf = pd.Series(stats.acf(ts.dropna(), nlags=len(ts)//4))
            seasonal_periods = acf.iloc[1:].nlargest(3)
            
            analysis["seasonality"] = {
                "top_periods": {
                    f"period_{i+1}": {
                        "lag": int(period.index),
                        "correlation": period.item()
                    }
                    for i, period in enumerate(seasonal_periods)
                }
            }
            
        return self.convert_to_json_serializable(analysis)

    def feature_importance(self, 
                         target_column: str, 
                         method: str = 'correlation') -> Dict[str, Any]:
        """Calculate feature importance relative to a target column"""
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
            
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if method == 'correlation':
            # Use absolute correlation values as importance scores
            correlations = numeric_df.corr()[target_column].abs()
            correlations = correlations.sort_values(ascending=False)
            
            result = {
                "method": "correlation",
                "importance_scores": correlations.to_dict()
            }

            return self.convert_to_json_serializable(result)
        
        elif method == 'pca':
            # Use PCA to analyze feature importance
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            
            pca = PCA()
            pca.fit(scaled_data)
            
            # Calculate feature importance based on component weights
            feature_importance = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(len(pca.components_))],
                index=numeric_df.columns
            )
            
            result = {
                "method": "pca",
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_).tolist(),
                "feature_weights": feature_importance.to_dict()
            }
            return self.convert_to_json_serializable(result)
            
        return {"error": "Unsupported method specified"}

    def clustering_analysis(self, 
                          columns: List[str], 
                          n_clusters: int = 3) -> Dict[str, Any]:
        """Perform clustering analysis on specified columns"""
        if not all(col in self.df.columns for col in columns):
            raise ValueError("One or more specified columns not found in DataFrame")
            
        # Prepare data for clustering
        data = self.df[columns].select_dtypes(include=[np.number])
        
        # Standardize the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_mask = clusters == i
            cluster_stats[f"cluster_{i}"] = {
                "size": sum(cluster_mask),
                "percentage": (sum(cluster_mask) / len(clusters)) * 100,
                "centroid": {
                    col: val for col, val in zip(
                        data.columns, 
                        kmeans.cluster_centers_[i]
                    )
                },
                "feature_means": data[cluster_mask].mean().to_dict(),
                "feature_stds": data[cluster_mask].std().to_dict()
            }
            
        result = {
            "n_clusters": n_clusters,
            "cluster_stats": cluster_stats,
            "inertia": kmeans.inertia_,
            "cluster_assignments": clusters.tolist()
        }

        return self.convert_to_json_serializable(result)
    

    def execute_code(self, code: str) -> str:
        import subprocess
        import tempfile
        import pandas as pd

        # Save the dataframe to a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_df_file:
            self.df.to_csv(temp_df_file.name, index=False)
            temp_df_path = temp_df_file.name

        # Prepare the execution script
        execution_script = f'''
        import pandas as pd

        # Load the dataframe
        df = pd.read_csv('{temp_df_path}')

        # Execute the user code
        {code}

        # If there's a result to return, it should be in a variable named 'result'
        try:
            print(result)
        except NameError:
            pass
        '''

        # Write the execution script to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_code_file:
            temp_code_file.write(execution_script)
            temp_code_path = temp_code_file.name

        # Execute the script and capture output
        try:
            output = subprocess.check_output(['python', temp_code_path], stderr=subprocess.STDOUT, text=True)
        except subprocess.CalledProcessError as e:
            output = f"An error occurred: {e.output}"
        finally:
            # Remove temporary files
            import os
            os.remove(temp_df_path)
            os.remove(temp_code_path)

        return output.strip()

# Function definitions for tool calling
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_column_stats",
            "description": "Calculate comprehensive statistics for a specific column in the dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "column_name": {
                        "type": "string",
                        "description": "Name of the column to analyze"
                    }
                },
                "required": ["column_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "correlation_analysis",
            "description": "Analyze correlations between numeric columns in the dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of columns to analyze (optional, defaults to all numeric columns)"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["pearson", "spearman", "kendall"],
                        "description": "Correlation method to use"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "distribution_analysis",
            "description": "Analyze the distribution of a column, including normality tests for numeric data",
            "parameters": {
                "type": "object",
                "properties": {
                    "column_name": {
                        "type": "string",
                        "description": "Name of the column to analyze"
                    }
                },
                "required": ["column_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "outlier_detection",
            "description": "Detect outliers in a numeric column using various statistical methods",
            "parameters": {
                "type": "object",
                "properties": {
                    "column_name": {
                        "type": "string",
                        "description": "Name of the column to analyze"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["zscore", "iqr"],
                        "description": "Method to use for outlier detection"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Threshold for outlier detection (e.g., 3.0 for z-score method)",
                        "default": 3.0
                    }
                },
                "required": ["column_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "time_series_analysis",
            "description": "Analyze time series data including trends and seasonality",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_column": {
                        "type": "string",
                        "description": "Name of the column containing dates"
                    },
                    "value_column": {
                        "type": "string",
                        "description": "Name of the column containing values to analyze"
                    },
                    "freq": {
                        "type": "string",
                        "description": "Frequency for resampling (e.g., 'D' for daily, 'M' for monthly)",
                        "default": "D"
                    }
                },
                "required": ["date_column", "value_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "feature_importance",
            "description": "Calculate feature importance relative to a target column",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_column": {
                        "type": "string",
                        "description": "Name of the target column"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["correlation", "pca"],
                        "description": "Method to use for calculating feature importance"
                    }
                },
                "required": ["target_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clustering_analysis",
            "description": "Perform clustering analysis on specified columns",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of columns to use for clustering"
                    },
                    "n_clusters": {
                        "type": "integer",
                        "description": "Number of clusters to form",
                        "default": 3
                    }
                },
                "required": ["columns"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code using the dataframe. Use this for custom analysis or visualizations. Make sure your code always returns an output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    }
]