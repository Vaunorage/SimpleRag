# SimpleRAG

**SimpleRAG** is a comprehensive document indexing and retrieval system built with Python. It leverages DuckDB for efficient storage and querying, and integrates with `llama-index` for state-of-the-art document processing and embedding. This toolkit is designed to handle everything from document ingestion and chunking to advanced search functionalities, including BM25, vector, and hybrid search.

## Features

-   **DuckDB Backend**: Uses DuckDB for a fast, file-based, and feature-rich database.
-   **Document Processing**: Ingests and processes a variety of document formats (`.pdf`, `.docx`, `.csv`, `.md`, and more).
-   **Text Chunking**: Splits documents into manageable text nodes using `SentenceSplitter`.
-   **Embedding Generation**: Creates vector embeddings for text nodes using models like OpenAI's, with built-in retry logic for rate limiting.
-   **Advanced Search**:
    -   **BM25 Search**: Efficient full-text search.
    -   **Vector Search**: Semantic search using HNSW for fast nearest neighbor lookup.
    -   **Hybrid Search**: Combines BM25 and vector search scores for improved relevance.
    -   **LIKE Search**: Simple SQL `LIKE` search.
-   **VLM-powered Document Loading**: Utilizes Vision Language Models (VLMs) for advanced OCR and document understanding, with support for local models (LM Studio, Ollama) and cloud services (IBM WatsonX).
-   **Incremental Indexing**: Filters out already processed documents and nodes to avoid redundant work.
-   **Data Integrity**: Removes outdated documents from the index based on an `update_date` metadata field.

## Installation

To install the package, clone the repository and use `pip` to install it in editable mode. This is recommended for development as it allows your code changes to be reflected immediately without reinstalling.

```bash
# Clone the repository (update the URL to your repository)
git clone https://github.com/your-username/SimpleRag.git
cd SimpleRag

# Install in editable mode
pip install -e .
```

This command installs `simplerag` and all its dependencies as defined in `pyproject.toml`.

## Setup

### Environment Variables

For using OpenAI embeddings or IBM WatsonX, you need to set up environment variables. Create a `.env` file in the root of your project:

```
# For OpenAI Embeddings
OPENAI_API_KEY="your-openai-api-key"

# For IBM WatsonX
WX_API_KEY="your-watsonx-api-key"
WX_PROJECT_ID="your-watsonx-project-id"
```

The `SimpleRAG` class and `watsonx_vlm_options` function will automatically load these variables.

## Usage

### 1. Initialize the Index

First, create an instance of `SimpleRAG`. You can configure parameters like the database path, embedding model, and chunking strategy.

```python
from simplerag import SimpleRAG

# Initialize the index
rag_index = SimpleRAG(
    db_path="my_local_index.db",
    use_embeddings=True  # Set to False to disable embeddings
)
```

### 2. Load Documents

Use the `load_documents_from_folder` function to load documents from a specified directory. You can choose a processing type: `'simple'`, `'vlm'`, or `'watsonx'`.

```python
from simplerag import load_documents_from_folder

# Load documents using simple processing
documents = load_documents_from_folder(
    folder_path="./path/to/your/documents",
    processing_type='simple'
)
```

### 3. Process and Index Documents

Pass the loaded documents to the `process_documents` method. This will create nodes, generate embeddings (if enabled), and write them to the database.

```python
rag_index.process_documents(documents)
```

### 4. Search the Index

The `search` method provides a unified interface for all search types. You can specify the `search_type` as `'hybrid'`, `'bm25'`, `'vector'`, or `'like'`.

```python
# Perform a hybrid search
results = rag_index.search(
    query="what is the main topic?",
    search_type="hybrid",
    limit=5
)

for node in results:
    print(f"Score: {node.extra_info['score']:.4f}")
    print(node.get_text())
    print("-----")
```

## Advanced Usage

### Document Loading with VLM

For more advanced document processing, especially for PDFs with complex layouts, you can use a Vision Language Model (VLM). The `load_documents_from_folder` function supports different VLM backends through the `processing_type` parameter.

-   **Local VLM (e.g., LM Studio)**: Use `processing_type='vlm'`.

    ```python
    documents = load_documents_from_folder(
        folder_path="./pdfs",
        processing_type='vlm'
    )
    ```

-   **IBM WatsonX**: Use `processing_type='watsonx'`. Make sure your `WX_API_KEY` and `WX_PROJECT_ID` are set in your `.env` file.

    ```python
    documents = load_documents_from_folder(
        folder_path="./pdfs",
        processing_type='watsonx'
    )
    ```

### Filtering and Exclusion

You can apply metadata filters and exclude specific node IDs from your search results.

```python
# Search with a metadata filter
filtered_results = rag_index.search(
    query="search query",
    search_type="hybrid",
    filters={"category": "finance"} # Assumes 'category' is in your document metadata
)

# Exclude certain nodes from the search
excluded_results = rag_index.search(
    query="search query",
    search_type="hybrid",
    exclude_ids=["node_id_1", "node_id_2"]
)
```

### Database Management

If you need to start fresh or fix a corrupted database, you can recreate it.

```python
# Recreate the database, backing up the old one
rag_index.recreate_database(backup_old=True)
```