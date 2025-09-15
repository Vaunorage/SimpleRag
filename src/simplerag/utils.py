import pandas as pd
import tiktoken


def count_tokens(text):
    """Count tokens using OpenAI's tiktoken tokenizer."""
    if pd.isna(text) or text == '':
        return 0

    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(str(text)))