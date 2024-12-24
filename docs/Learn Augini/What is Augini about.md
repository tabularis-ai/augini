# What is Augini about?

`augini` enables synthetic data generation in an easy to use interface.

Some datasets are hard to get access to. Synthetic data generation allows one to run quick experiments on datasets that are very similar to ground truth. 

The tool currently exposes three easy to use interfaces/methods to get to work quickly.

### User-Facing Methods:

#### `augment_columns`:

Purpose: This method is used to augment (add new synthetic data to) specific columns in a DataFrame.

Usage: Users can specify which columns/column names they want to augment and optionally provide a custom prompt. 

Example:

```python
augmented_df = augini.augment_columns(df, columns=["column1", "column2"], custom_prompt="Custom prompt here", use_sync=False)
```

Columns can be of arbitrary size. That is, they can be more than the two presented above.

#### `augment_single`:

Purpose: This method is used to augment a single column in a DataFrame.

Usage: Similar to augment_columns, but specifically for a single column. 

Example:

```python
augmented_df = augini.augment_single(df, column_name="column1", custom_prompt="Custom prompt here", use_sync=False)
print(augmented_df.head())
```

#### `chat`:

Purpose: This method allows users to ask natural language questions about a DataFrame and receive AI-generated responses.

Usage: Users provide a query and the DataFrame, and the method returns an AI-generated answer based on the DataFrame's context.

Example:

```python
response = augini.chat(query="What is the average age in the DataFrame?", df=df)
print(response)
```

You can ask follow up questions by simply querying augini again.

```python
response = augini.chat(query="What is the average presented in this case the median or the mean?", df=df)
print(response)
```

### Chat oriented methods:

#### `get_conversation_history`:

Purpose: Retrieves the conversation history in either 'full' or 'summary' mode.

Usage: Users can get an overview of the current conversation context. It can be helpful to know what data and conversation details `augini` will be interacting with.

Example:

```python
full_history = augini.get_conversation_history(mode='full')
print(full_history)

summary_history = augini.get_conversation_history(mode='summary')
print(summary_history)
```

The full mode presents an overview of the entire conversation. Summaries make use of embeddings to aim to reduce the cost of api queries ensuring that only contextually relevant information is used in any query.

#### `clear_conversation_history`:

Purpose:Clears the conversation history based on the specified mode.

Usage: By clearing the conversation history you start from a new slate making it less likely that a previously error filled context affects new instructions thereby increasing accuracy.

Example:

```python
# Clear full conversation history
augini.clear_conversation_history(mode='full')

# Clear summarized context histories
augini.clear_conversation_history(mode='summary')

# Clear both
augini.clear_conversation_history(mode='all')
```