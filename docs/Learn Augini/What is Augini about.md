# What is Augini about?

Augini enables synthetic data generation in an easy to use interface.

Some datasets are hard to get access to. Synthetic data generation allows one to run quick experiments on
datasets that are very similar to ground truth. 

The tool currently exposes three easy to use interfaces/methods to get to work quickly.


### User-Facing Methods:

#### `augment_columns`:

Purpose: This method is used to augment (add new synthetic data to) specific columns in a DataFrame.

Usage: Users can specify which columns they want to augment and optionally provide a custom prompt. The method can either use asynchronous or synchronous processing.

Example:

```python
augmented_df = augini_instance.augment_columns(df, columns=["column1", "column2"], custom_prompt="Custom prompt here", use_sync=False)
augment_single:
```

#### `augment_single`:

Purpose: This method is used to augment a single column in a DataFrame.

Usage: Similar to augment_columns, but specifically for a single column. Users can provide a custom prompt and choose between asynchronous or synchronous processing.

Example:

```python
augmented_df = augini_instance.augment_single(df, column_name="column1", custom_prompt="Custom prompt here", use_sync=False)
```

#### `chat`:

Purpose: This method allows users to ask natural language questions about a DataFrame and receive AI-generated responses.

Usage: Users provide a query and the DataFrame, and the method returns an AI-generated answer based on the DataFrame's context.

Example:

```python
response = augini_instance.chat(query="What is the average age in the DataFrame?", df=df)
print(response)
```

### Chat oriented methods:

#### `get_conversation_history`:

Purpose: Retrieves the conversation history in either 'full' or 'summary' mode.

Example:

```python
full_history = augini_instance.get_conversation_history(mode='full')
summary_history = augini_instance.get_conversation_history(mode='summary')
```

#### `clear_conversation_history`:

Purpose:Clears the conversation history based on the specified mode.

Example:

```python
# Clear full conversation history
augini_instance.clear_conversation_history(mode='full')

# Clear summarized context histories
augini_instance.clear_conversation_history(mode='summary')

# Clear both
augini_instance.clear_conversation_history(mode='all')
```

Summaries aid speed and reduce the cost of api queries ensuring that only contextually relevant information is used in any query.