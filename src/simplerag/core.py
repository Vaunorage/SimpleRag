import asyncio
import json
import os
import random
import re
import shutil
import time
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import duckdb
import pandas as pd
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.utils import count_tokens
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import RateLimitError
from tqdm import tqdm
from dotenv import load_dotenv

from docling_core.types.doc.page import SegmentedPage
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, PdfFormatOption, CsvFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel import vlm_model_specs


class SimpleRAG:
    """
    A comprehensive document index using DuckDB, handling everything from
    document processing and embedding to advanced search functionalities like
    BM25, vector search, and hybrid search.
    """

    def __init__(
            self,
            db_path: str,
            embedding_dim: int = 1536,
            embed_model: Optional[object] = None,
            use_embeddings: bool = True,
            batch_size: int = 50,
            chunk_size: int = 1024,
            chunk_overlap: int = 50,
            delay_between_batches: float = 2.0,
            base_retry_delay: float = 10.0,
            jitter: float = 1.0,
    ) -> None:
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.use_embeddings = use_embeddings

        # Set up the embedding model
        if embed_model:
            self.embed_model = embed_model
        else:
            Settings.embed_model = OpenAIEmbedding()
            self.embed_model = Settings.embed_model

        # Document processing settings
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.delay_between_batches = delay_between_batches
        self.base_retry_delay = base_retry_delay
        self.jitter = jitter

        # Initialize database
        self._setup_database_conn()
        self._setup_extensions()
        self._setup_tables()
        self._setup_indexes()

    # =================================================================================
    # Document Processing and Indexing
    # =================================================================================

    def process_documents(self, documents: List[Document]):
        """Main processing function to index a list of documents."""
        print(f"Starting document processing (embeddings: {'enabled' if self.use_embeddings else 'disabled'})...")
        start_time = time.time()

        self.remove_outdated_documents(documents)
        nodes = self._create_nodes_from_documents(documents)
        nodes = self._filter_existing_nodes(nodes)

        if not nodes:
            print("No new nodes to process!")
            return

        total_processed = 0
        total_written = 0

        if self.use_embeddings:
            batches = [nodes[i:i + self.batch_size] for i in range(0, len(nodes), self.batch_size)]
            print(f"Created {len(batches)} batches for embedding processing")

            with tqdm(total=len(nodes), desc="Processing") as pbar:
                for i, batch in enumerate(batches):
                    print(f"\nProcessing batch {i + 1}/{len(batches)}")
                    embedded_batch = self._embed_batch_with_retry(batch)

                    written_count = self.update_nodes(
                        embedded_batch,
                        preserve_order=True,
                        skip_existing=True
                    )

                    total_processed += len(embedded_batch)
                    total_written += written_count
                    pbar.update(len(embedded_batch))

                    if i < len(batches) - 1:
                        delay = self.delay_between_batches + random.uniform(0, self.jitter)
                        print(f"Waiting {delay:.2f}s before next batch...")
                        time.sleep(delay)
        else:
            print(f"Writing {len(nodes)} nodes to database (no embeddings)...")
            written_count = self.update_nodes(nodes, preserve_order=True, skip_existing=True)
            total_processed = len(nodes)
            total_written = written_count

        total_time = time.time() - start_time
        print("\n=== Processing Complete ===")
        print(f"Total nodes processed: {total_processed}")
        print(f"Total nodes written: {total_written}")
        print(f"Embeddings created: {'Yes' if self.use_embeddings else 'No'}")
        print(f"Total time: {total_time:.2f}s")

    def _create_nodes_from_documents(self, documents: List[Document]) -> List[BaseNode]:
        """Create nodes from a list of documents."""
        print("Creating nodes from documents...")
        splitter = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
        print(f"Created {len(nodes)} total nodes")
        return nodes

    def _filter_existing_nodes(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """Filter out nodes that already exist in the database based on hash."""
        if not nodes:
            return []

        existing_hashes = set(self.get_all_node_hashes())
        if not existing_hashes:
            return nodes

        filtered_nodes = [node for node in nodes if node.hash not in existing_hashes]
        print(f"Filtered out {len(nodes) - len(filtered_nodes)} existing nodes")
        return filtered_nodes

    def _embed_batch_with_retry(self, batch: List[BaseNode]) -> List[BaseNode]:
        """Embed a batch of nodes with retry logic for rate limiting."""
        texts = [node.get_text() for node in batch]

        async def _embed_async():
            max_retries = 10
            for i in range(max_retries):
                try:
                    return await self.embed_model.aget_text_embedding_batch(texts)
                except RateLimitError as e:
                    error_message = str(e)
                    match = re.search(r'retry after (\d+) seconds', error_message)
                    wait_time = int(match.group(1)) if match else self.base_retry_delay

                    print(f"Rate limit hit. Retrying in {wait_time}s... (Attempt {i + 1}/{max_retries})")
                    await asyncio.sleep(wait_time + self.jitter)

            print("Max retries exceeded for embedding. Skipping batch.")
            return [None] * len(batch)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            embeddings = loop.run_until_complete(_embed_async())
            for node, embedding in zip(batch, embeddings):
                if embedding:
                    node.embedding = embedding
        finally:
            loop.close()

        return batch

    # =================================================================================
    # Database Setup and Management
    # =================================================================================

    def _setup_database_conn(self):
        """Establish a connection to the DuckDB database."""
        try:
            self.conn = duckdb.connect(self.db_path)
        except Exception as e:
            print(f"Error connecting to database: {e}. Attempting to recreate.")
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.conn = duckdb.connect(self.db_path)

    def _setup_extensions(self):
        """Install and load required DuckDB extensions."""
        self.conn.execute("INSTALL fts;")
        self.conn.execute("LOAD fts;")
        if self.use_embeddings:
            self.conn.execute("INSTALL vss;")
            self.conn.execute("LOAD vss;")
            self.conn.execute("SET hnsw_enable_experimental_persistence = true;")

    def _setup_tables(self):
        """Create database tables if they don't exist."""
        create_docs_sql = f"""
        CREATE TABLE IF NOT EXISTS documents (
            node_id VARCHAR PRIMARY KEY,
            ref_doc_id VARCHAR NOT NULL,
            text VARCHAR NOT NULL,
            extra_data VARCHAR,
            embedding FLOAT[{self.embedding_dim}],
            token_count INTEGER,
            hash VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.conn.execute(create_docs_sql)

        create_order_sql = """
        CREATE TABLE IF NOT EXISTS node_order (
            ref_doc_id VARCHAR NOT NULL,
            node_id VARCHAR NOT NULL,
            node_order INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            PRIMARY KEY (ref_doc_id, node_id),
            FOREIGN KEY (node_id) REFERENCES documents(node_id)
        );
        """
        self.conn.execute(create_order_sql)

    def _setup_indexes(self):
        """Create FTS and HNSW indexes for efficient searching."""
        try:
            # Drop existing FTS index to avoid errors on re-initialization
            self.conn.execute("PRAGMA drop_fts_index('documents');")
        except duckdb.CatalogException:
            # This is expected if the index doesn't exist yet
            pass
        except Exception as e:
            print(f"Warning: Could not drop FTS index: {e}")

        try:
            self.conn.execute("PRAGMA create_fts_index('documents', 'node_id', 'text', overwrite=1);")
        except Exception as e:
            print(f"Error creating FTS index: {e}")

        if self.use_embeddings:
            try:
                count = self.conn.execute("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL").fetchone()[0]
                if count > 0:
                    self.conn.execute(
                        "CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx ON documents USING HNSW (embedding);")
            except Exception as e:
                print(f"Warning: Vector index creation failed: {e}")

    def recreate_database(self, backup_old: bool = True):
        """Recreates the database, useful for fixing corruption or schema changes."""
        print("Recreating database...")
        self.close()

        if backup_old and os.path.exists(self.db_path):
            backup_path = f"{self.db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(self.db_path, backup_path)
            print(f"Backed up old database to: {backup_path}")
        elif os.path.exists(self.db_path):
            os.remove(self.db_path)

        # Re-initialize
        self._setup_database_conn()
        self._setup_extensions()
        self._setup_tables()
        self._setup_indexes()
        print("Database recreated successfully.")

    # =================================================================================
    # CRUD Operations on Nodes
    # =================================================================================

    def update_nodes(self, nodes: List[BaseNode], preserve_order: bool = True, skip_existing: bool = False):
        """Update or add multiple nodes to the index."""
        if not nodes:
            return 0

        node_dicts = [self._node_to_dict(node) for node in nodes]

        if skip_existing:
            hashes = [nd['hash'] for nd in node_dicts if nd['hash']]
            if hashes:
                placeholders = ', '.join(['?'] * len(hashes))
                existing_hashes = {row[0] for row in
                                   self.conn.execute(f"SELECT hash FROM documents WHERE hash IN ({placeholders})",
                                                     hashes).fetchall()}
                node_dicts = [nd for nd in node_dicts if nd['hash'] not in existing_hashes]

        if not node_dicts:
            return 0

        df = pd.DataFrame(node_dicts)

        # Use DuckDB's native DataFrame insertion
        self.conn.execute(
            f"INSERT OR REPLACE INTO documents (node_id, ref_doc_id, text, extra_data, embedding, token_count, hash, created_at, updated_at) SELECT * FROM df")

        if preserve_order:
            order_data = []
            for ref_doc_id, group in df.groupby('ref_doc_id'):
                for i, row in group.reset_index(drop=True).iterrows():
                    order_data.append((ref_doc_id, row['node_id'], i, row['token_count']))

            if order_data:
                self.conn.executemany("INSERT OR REPLACE INTO node_order VALUES (?, ?, ?, ?)", order_data)

        self._refresh_indexes()
        return len(node_dicts)

    def remove_document(self, ref_doc_id: str):
        """Remove all nodes associated with a specific document ID."""
        print(f"Removing document: {ref_doc_id}...")
        self.conn.execute("DELETE FROM node_order WHERE ref_doc_id = ?", [ref_doc_id])
        self.conn.execute("DELETE FROM documents WHERE ref_doc_id = ?", [ref_doc_id])
        self._refresh_indexes()
        print("Removal complete.")

    def remove_outdated_documents(self, documents: List[Document]):
        """Removes documents from the index if the new version has a more recent 'update_date'."""
        docs_by_id = {doc.doc_id: doc for doc in documents if hasattr(doc, 'doc_id') and doc.doc_id}
        if not docs_by_id:
            return

        placeholders = ', '.join(['?'] * len(docs_by_id))
        query = f"SELECT ref_doc_id, extra_data FROM documents WHERE ref_doc_id IN ({placeholders})"

        try:
            results = self.conn.execute(query, list(docs_by_id.keys())).fetchall()

            docs_to_remove = []
            for ref_doc_id, extra_data_json in results:
                current_doc = docs_by_id.get(ref_doc_id)
                if not current_doc:
                    continue

                current_update_date = current_doc.extra_info.get('update_date')

                try:
                    existing_update_date = json.loads(extra_data_json).get('update_date')
                except:
                    existing_update_date = None

                if current_update_date and (
                        not existing_update_date or str(current_update_date) > str(existing_update_date)):
                    docs_to_remove.append(ref_doc_id)

            if docs_to_remove:
                print(f"Found {len(docs_to_remove)} outdated documents to remove.")
                for doc_id in docs_to_remove:
                    self.remove_document(doc_id)

        except Exception as e:
            print(f"Could not check for outdated documents: {e}")

    # =================================================================================
    # Search and Retrieval
    # =================================================================================

    def search(
            self,
            query: str,
            limit: int = 10,
            search_type: str = "hybrid",
            filters: Optional[Dict[str, Any]] = None,
            exclude_ids: Optional[List[str]] = None,
    ) -> List[BaseNode]:
        """
        A unified search function.

        Args:
            query: The search query string.
            limit: Number of results to return.
            search_type: 'bm25', 'vector', 'like', or 'hybrid'.
            filters: Metadata filters to apply.
            exclude_ids: Node IDs to exclude from results.
        """
        if search_type == "hybrid":
            return self.search_hybrid(query, limit=limit, filters=filters, exclude_ids=exclude_ids)
        elif search_type == "bm25":
            return self.search_by_bm25(query, limit=limit, filters=filters, exclude_ids=exclude_ids)
        elif search_type == "vector":
            if not self.use_embeddings:
                raise ValueError("Cannot perform vector search when use_embeddings is False.")
            return self.search_by_vector(query, limit=limit, filters=filters, exclude_ids=exclude_ids)
        elif search_type == "like":
            return self.search_by_like(query, limit=limit, filters=filters, exclude_ids=exclude_ids)
        else:
            raise ValueError(f"Unknown search_type: {search_type}")

    def search_hybrid(self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None,
                      exclude_ids: Optional[List[str]] = None) -> List[BaseNode]:
        """Performs a hybrid search combining BM25 and vector scores."""
        params = []

        # BM25 Score
        bm25_score_sql = "COALESCE(fts_main_documents.match_bm25(node_id, ?), 0)"
        params.append(query)

        # Vector Score (Similarity)
        if self.use_embeddings:
            query_embedding = self.embed_model.get_text_embedding(query)
            embedding_str = f"[{','.join(map(str, query_embedding))}]::FLOAT[{self.embedding_dim}]"
            vector_score_sql = f"(1.0 / (1.0 + array_distance(embedding, {embedding_str})))"
        else:
            vector_score_sql = "0"

        # Unified score with weights
        unified_score_sql = f"({bm25_score_sql} * 1.0) + ({vector_score_sql} * 100.0) AS score"

        # Build query
        sql = f"SELECT *, {unified_score_sql} FROM documents"
        sql, filter_params = self._apply_filters_to_sql(sql, filters)
        params.extend(filter_params)

        if exclude_ids:
            conjunction = "WHERE" if "WHERE" not in sql else "AND"
            placeholders = ",".join("?" * len(exclude_ids))
            sql += f" {conjunction} node_id NOT IN ({placeholders})"
            params.extend(exclude_ids)

        sql += " ORDER BY score DESC LIMIT ?"
        params.append(limit)

        results = self.conn.execute(sql, params).fetchall()
        return [self._row_to_node(row, self.conn.description) for row in results]

    def search_by_bm25(self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None,
                       exclude_ids: Optional[List[str]] = None) -> List[BaseNode]:
        """Search using BM25 full-text search."""
        params = [query]
        sql = "SELECT *, fts_main_documents.match_bm25(node_id, ?) AS score FROM documents WHERE score IS NOT NULL"

        sql, filter_params = self._apply_filters_to_sql(sql, filters, "AND")
        params.extend(filter_params)

        if exclude_ids:
            placeholders = ",".join("?" * len(exclude_ids))
            sql += f" AND node_id NOT IN ({placeholders})"
            params.extend(exclude_ids)

        sql += " ORDER BY score DESC LIMIT ?"
        params.append(limit)

        results = self.conn.execute(sql, params).fetchall()
        return [self._row_to_node(row, self.conn.description) for row in results]

    def search_by_vector(self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None,
                         exclude_ids: Optional[List[str]] = None) -> List[BaseNode]:
        """Search using vector similarity."""
        query_embedding = self.embed_model.get_text_embedding(query)
        embedding_str = f"[{','.join(map(str, query_embedding))}]::FLOAT[{self.embedding_dim}]"

        params = []
        sql = f"SELECT *, array_distance(embedding, {embedding_str}) AS distance FROM documents WHERE embedding IS NOT NULL"

        sql, filter_params = self._apply_filters_to_sql(sql, filters, "AND")
        params.extend(filter_params)

        if exclude_ids:
            placeholders = ",".join("?" * len(exclude_ids))
            sql += f" AND node_id NOT IN ({placeholders})"
            params.extend(exclude_ids)

        sql += " ORDER BY distance ASC LIMIT ?"
        params.append(limit)

        results = self.conn.execute(sql, params).fetchall()
        return [self._row_to_node(row, self.conn.description) for row in results]

    def search_by_like(self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None,
                       exclude_ids: Optional[List[str]] = None) -> List[BaseNode]:
        """Search using SQL LIKE operator."""
        pattern = f'%{query.lower()}%'
        params = [pattern, pattern]

        sql = "SELECT * FROM documents WHERE LOWER(text) LIKE ? OR LOWER(extra_data) LIKE ?"

        sql, filter_params = self._apply_filters_to_sql(sql, filters, "AND")
        params.extend(filter_params)

        if exclude_ids:
            placeholders = ",".join("?" * len(exclude_ids))
            sql += f" AND node_id NOT IN ({placeholders})"
            params.extend(exclude_ids)

        sql += " LIMIT ?"
        params.append(limit)

        results = self.conn.execute(sql, params).fetchall()
        return [self._row_to_node(row, self.conn.description) for row in results]

    # =================================================================================
    # Utility and Helper Methods
    # =================================================================================

    def _node_to_dict(self, node: BaseNode) -> Dict[str, Any]:
        """Convert a LlamaIndex BaseNode to a dictionary for database insertion."""
        text = node.get_text() or ''
        extra_data = {**(node.metadata or {}), **(node.extra_info or {})}

        current_time = datetime.now()
        return {
            'node_id': node.node_id,
            'ref_doc_id': node.ref_doc_id,
            'text': text,
            'extra_data': json.dumps(extra_data),
            'embedding': node.embedding if hasattr(node, 'embedding') and node.embedding else None,
            'token_count': count_tokens(text),
            'hash': node.hash,
            'created_at': current_time,
            'updated_at': current_time,
        }

    def _row_to_node(self, row: tuple, description: list) -> BaseNode:
        """Convert a database row tuple to a LlamaIndex TextNode."""
        row_dict = {desc[0]: val for desc, val in zip(description, row)}
        extra_info = json.loads(row_dict.get('extra_data') or '{}')

        node = TextNode(
            id_=row_dict['node_id'],
            text=row_dict['text'],
            embedding=row_dict.get('embedding'),
            extra_info=extra_info
        )
        node.extra_info['score'] = row_dict.get('score')
        node.extra_info['distance'] = row_dict.get('distance')
        return node

    def _apply_filters_to_sql(self, sql: str, filters: Optional[Dict[str, Any]], conjunction: str = "WHERE") -> tuple[
        str, list]:
        """Applies metadata filters to a SQL query."""
        if not filters:
            return sql, []

        conditions = []
        params = []
        for key, value in filters.items():
            conditions.append("json_extract_string(extra_data, ?) = ?")
            params.extend([f"$.{key}", str(value)])

        if conditions:
            sql += f" {conjunction} " + " AND ".join(conditions)

        return sql, params

    def _refresh_indexes(self):
        """Refresh FTS and vector indexes after data modification."""
        try:
            self.conn.execute("PRAGMA drop_fts_index('documents');")
            self.conn.execute("PRAGMA create_fts_index('documents', 'node_id', 'text');")
        except Exception as e:
            print(f"Warning: FTS index refresh failed: {e}")

        if self.use_embeddings:
            try:
                self.conn.execute("DROP INDEX IF EXISTS documents_embedding_hnsw_idx;")
                count = self.conn.execute("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL").fetchone()[0]
                if count > 0:
                    self.conn.execute("CREATE INDEX documents_embedding_hnsw_idx ON documents USING HNSW (embedding);")
            except Exception as e:
                print(f"Warning: Vector index refresh failed: {e}")

    def get_all_node_hashes(self) -> List[str]:
        """Retrieve all unique node hashes from the database."""
        try:
            return [row[0] for row in self.conn.execute("SELECT hash FROM documents WHERE hash IS NOT NULL").fetchall()]
        except Exception as e:
            print(f"Error getting node hashes: {e}")
            return []

    def close(self):
        """Close the database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def lms_vlm_options(model: str, prompt: str, format: ResponseFormat):
    options = ApiVlmOptions(
        url="http://localhost:1234/v1/chat/completions",  # the default LM Studio
        params=dict(
            model=model,
        ),
        prompt=prompt,
        timeout=90,
        scale=1.0,
        response_format=format,
    )
    return options


def lms_olmocr_vlm_options(model: str):
    class OlmocrVlmOptions(ApiVlmOptions):
        def build_prompt(self, page: Optional[SegmentedPage]) -> str:
            if page is None:
                return self.prompt.replace("#RAW_TEXT#", "")

            anchor = [
                f"Page dimensions: {int(page.dimension.width)}x{int(page.dimension.height)}"
            ]

            for text_cell in page.textline_cells:
                if not text_cell.text.strip():
                    continue
                bbox = text_cell.rect.to_bounding_box().to_bottom_left_origin(
                    page.dimension.height
                )
                anchor.append(f"[{int(bbox.l)}x{int(bbox.b)}] {text_cell.text}")

            for image_cell in page.bitmap_resources:
                bbox = image_cell.rect.to_bounding_box().to_bottom_left_origin(
                    page.dimension.height
                )
                anchor.append(
                    f"[Image {int(bbox.l)}x{int(bbox.b)} to {int(bbox.r)}x{int(bbox.t)}]"
                )

            if len(anchor) == 1:
                anchor.append(
                    f"[Image 0x0 to {int(page.dimension.width)}x{int(page.dimension.height)}]"
                )

            # Original prompt uses cells sorting. We are skipping it for simplicity.

            raw_text = "\n".join(anchor)

            return self.prompt.replace("#RAW_TEXT#", raw_text)

        def decode_response(self, text: str) -> str:
            # OlmOcr trained to generate json response with language, rotation and other info
            try:
                generated_json = json.loads(text)
            except json.decoder.JSONDecodeError:
                return ""

            return generated_json["natural_text"]

    options = OlmocrVlmOptions(
        url="http://localhost:1234/v1/chat/completions",
        params=dict(
            model=model,
        ),
        prompt=(
            "Below is the image of one page of a document, as well as some raw textual"
            " content that was previously extracted for it. Just return the plain text"
            " representation of this document as if you were reading it naturally.\n"
            "Do not hallucinate.\n"
            "RAW_TEXT_START\n#RAW_TEXT#\nRAW_TEXT_END"
        ),
        timeout=90,
        scale=1.0,
        max_size=1024,  # from OlmOcr pipeline
        response_format=ResponseFormat.MARKDOWN,
    )
    return options


def ollama_vlm_options(model: str, prompt: str):
    options = ApiVlmOptions(
        url="http://localhost:11434/v1/chat/completions",  # the default Ollama endpoint
        params=dict(
            model=model,
        ),
        prompt=prompt,
        timeout=90,
        scale=1.0,
        response_format=ResponseFormat.MARKDOWN,
    )
    return options


def watsonx_vlm_options(model: str, prompt: str):
    """Configures VLM options for IBM WatsonX.

    Note: This function requires WX_API_KEY and WX_PROJECT_ID to be set in the environment.
    If you encounter a 403 Forbidden error, please ensure that:
    1. Your API key and project ID are correct.
    2. The user associated with the API key has the 'Editor' or 'Admin' role in the WatsonX project.
    """
    load_dotenv()
    api_key = os.environ.get("WX_API_KEY")
    project_id = os.environ.get("WX_PROJECT_ID")

    def _get_iam_access_token(api_key: str) -> str:
        res = requests.post(
            url="https://iam.cloud.ibm.com/identity/token",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data=f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}",
        )
        res.raise_for_status()
        api_out = res.json()
        print(f"{api_out=}")
        return api_out["access_token"]

    options = ApiVlmOptions(
        url="https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29",
        params=dict(
            model_id=model,
            project_id=project_id,
            parameters=dict(
                max_new_tokens=400,
            ),
        ),
        headers={
            "Authorization": "Bearer " + _get_iam_access_token(api_key=api_key),
        },
        prompt=prompt,
        timeout=60,
        response_format=ResponseFormat.MARKDOWN,
    )
    return options


def load_documents_from_folder(folder_path: str, processing_type: str = 'simple',
                               doc_types: Optional[List[str]] = None) -> list[Document]:
    """Loads documents from a folder into a list of LlamaIndex Documents using different processing methods."""
    if doc_types is None:
        doc_types = ['.pdf', '.docx', '.xlsx', '.pptx', '.md', '.adoc', '.asciidoc', '.html', '.xhtml', '.csv', '.png',
                     '.jpeg', '.jpg', '.tiff', '.tif', '.bmp', '.webp']

    extension_to_format = {
        '.pdf': InputFormat.PDF,
        '.docx': InputFormat.DOCX,
        '.xlsx': InputFormat.XLSX,
        '.pptx': InputFormat.PPTX,
        '.md': InputFormat.MD,
        '.adoc': InputFormat.ASCIIDOC,
        '.asciidoc': InputFormat.ASCIIDOC,
        '.html': InputFormat.HTML,
        '.xhtml': InputFormat.HTML,
        '.csv': InputFormat.CSV,
        '.png': InputFormat.IMAGE,
        '.jpeg': InputFormat.IMAGE,
        '.jpg': InputFormat.IMAGE,
        '.tiff': InputFormat.IMAGE,
        '.tif': InputFormat.IMAGE,
        '.bmp': InputFormat.IMAGE,
        '.webp': InputFormat.IMAGE,
    }

    allowed_formats = list(set(extension_to_format[ext] for ext in doc_types if ext in extension_to_format))

    csv_format_option = CsvFormatOption(encoding="utf-8-sig", encoding_fallback="latin-1")

    documents = []

    if processing_type == 'simple':
        converter = DocumentConverter(
            allowed_formats=allowed_formats,
            format_options={InputFormat.CSV: csv_format_option}
        )
    elif processing_type == 'vlm':
        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_model_specs.SMOLDOCLING_MLX,
        )
        converter = DocumentConverter(
            allowed_formats=allowed_formats,
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                ),
                InputFormat.CSV: csv_format_option
            }
        )
    elif processing_type == 'watsonx':
        pipeline_options = VlmPipelineOptions(enable_remote_services=True)

        # Using WatsonX by default for remote VLM
        pipeline_options.vlm_options = watsonx_vlm_options(
            model="ibm/granite-vision-3-2-2b",
            prompt="OCR the full page to markdown."
        )

        converter = DocumentConverter(
            allowed_formats=allowed_formats,
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    pipeline_cls=VlmPipeline,
                ),
                InputFormat.CSV: csv_format_option
            }
        )
    else:
        raise ValueError(f"Unknown processing_type: {processing_type}")

    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                # Convert the document to a docling Document
                result = converter.convert(source=file_path)
                if result is None:  # converter skips non-allowed files
                    continue
                docling_doc = result.document
                # Export to markdown
                markdown_text = docling_doc.export_to_markdown()

                # Create a LlamaIndex Document object
                doc = Document(
                    text=markdown_text,
                    doc_id=filename,
                    extra_info={
                        "file_path": file_path,
                        "source": "local_folder",
                        "processing_type": processing_type
                    }
                )
                documents.append(doc)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Loaded {len(documents)} documents from '{folder_path}' using '{processing_type}' processing.")
    return documents
