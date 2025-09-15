
from llama_index.core.utils import count_tokens
from tqdm import tqdm
import asyncio
import time
import random
import re
import os
from openai import RateLimitError
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document, Settings
import pandas as pd
import duckdb
import json
from typing import List, Dict, Any, Optional
from llama_index.core.schema import BaseNode, TextNode

from src.index import DuckDBDocumentIndex


def nodes_to_dataframe(nodes: List[BaseNode]) -> pd.DataFrame:
    # Calculate token counts and prepare node data
    node_data = []
    for node in nodes:
        token_count = count_tokens(node.text or '')

        # Extract metadata and extra_info
        metadata = node.metadata or {}
        extra_info = node.extra_info or {}

        # Combine metadata and extra_info into a JSON string
        extra_data = json.dumps({
            "metadata": metadata,
            "extra_info": extra_info
        })

        node_data.append({
            "node_id": node.node_id,
            "ref_doc_id": node.ref_doc_id,
            "text": node.text,
            "extra_data": extra_data,
            "token_count": token_count,
            "hash": node.hash,
        })

    # Convert to DataFrame
    return pd.DataFrame(node_data)


def save_nodes_to_duckdb(df: pd.DataFrame, db_path: str, table_name: str = "documents",
                         preserve_order: bool = True, skip_existing: bool = False) -> int:
    """
    Save nodes from a DataFrame to a DuckDB database.

    Args:
        df: DataFrame containing node data
        db_path: Path to the DuckDB database
        table_name: Name of the table to save to
        preserve_order: Whether to preserve node order
        skip_existing: Whether to skip nodes that already exist

    Returns:
        int: Number of nodes saved
    """
    if df.empty:
        print("No nodes to save")
        return 0

    try:
        # Connect to DuckDB
        conn = duckdb.connect(db_path)

        # Create tables if they don't exist
        conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            node_id VARCHAR PRIMARY KEY,
            ref_doc_id VARCHAR NOT NULL,
            text VARCHAR NOT NULL,
            extra_data VARCHAR, -- JSON string
            token_count INTEGER,
            hash VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Create node_order table if preserving order
        if preserve_order:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS node_order (
                ref_doc_id VARCHAR NOT NULL,
                node_id VARCHAR NOT NULL,
                node_order INTEGER NOT NULL,
                token_count INTEGER,
                PRIMARY KEY (ref_doc_id, node_id)
            );
            """)

        # Filter out existing nodes if skip_existing is True
        if skip_existing and 'hash' in df.columns and not df['hash'].isna().all():
            hashes = df['hash'].dropna().tolist()

            if hashes:
                # Get existing hashes
                existing_hashes_df = conn.execute(
                    f"SELECT hash FROM {table_name} WHERE hash IN ({', '.join(['?'] * len(hashes))})",
                    hashes
                ).fetchdf()

                if not existing_hashes_df.empty:
                    existing_hashes = set(existing_hashes_df['hash'])
                    df = df[~df['hash'].isin(existing_hashes)]

        if df.empty:
            print("All nodes already exist in the database")
            conn.close()
            return 0

        # Insert nodes into documents table
        conn.execute(f"""
        INSERT OR REPLACE INTO {table_name} 
        (node_id, ref_doc_id, text, extra_data, token_count, hash, updated_at)
        SELECT node_id, ref_doc_id, text, extra_data, token_count, hash, CURRENT_TIMESTAMP
        FROM df
        """)

        # Insert node order information if preserve_order is True
        if preserve_order:
            # Create a DataFrame with order information
            order_data = []
            for ref_doc_id, group in df.groupby('ref_doc_id'):
                for i, (_, row) in enumerate(group.iterrows()):
                    order_data.append({
                        'ref_doc_id': ref_doc_id,
                        'node_id': row['node_id'],
                        'node_order': i,
                        'token_count': row['token_count']
                    })

            if order_data:
                order_df = pd.DataFrame(order_data)
                conn.execute("""
                INSERT OR REPLACE INTO node_order (ref_doc_id, node_id, node_order, token_count)
                SELECT ref_doc_id, node_id, node_order, token_count
                FROM order_df
                """)

        # Refresh FTS index if it exists
        try:
            conn.execute(f"REFRESH FTS INDEX fts_idx;")
        except:
            # FTS index might not exist, which is fine
            pass

        conn.close()
        return len(df)

    except Exception as e:
        print(f"Error saving nodes to DuckDB: {e}")
        return 0


def pandas_filter(nodes: List[BaseNode], db_path: str, table_name: str = "documents") -> List[BaseNode]:
    """
    Filter out nodes that already exist in the database based on their hash.

    Args:
        nodes: List of BaseNode objects
        db_path: Path to the DuckDB database
        table_name: Name of the table to check

    Returns:
        List[BaseNode]: Filtered list of nodes
    """
    if not nodes:
        return []

    # Extract hashes from nodes
    node_hashes = {node.hash: node for node in nodes if node.hash is not None}

    if not node_hashes:
        return nodes

    try:
        # Connect to DuckDB
        conn = duckdb.connect(db_path)

        # Check if table exists
        table_exists = conn.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'").fetchone()
        if not table_exists:
            conn.close()
            return nodes

        # Get existing hashes
        hashes = list(node_hashes.keys())
        placeholders = ', '.join(['?'] * len(hashes))
        existing_hashes_df = conn.execute(
            f"SELECT hash FROM {table_name} WHERE hash IN ({placeholders})",
            hashes
        ).fetchdf()

        conn.close()

        if existing_hashes_df.empty:
            return nodes

        # Filter out nodes with existing hashes
        existing_hashes = set(existing_hashes_df['hash'])
        filtered_nodes = [node for node in nodes if node.hash is None or node.hash not in existing_hashes]

        print(f"Filtered out {len(nodes) - len(filtered_nodes)} existing nodes")
        return filtered_nodes

    except Exception as e:
        print(f"Error checking for existing nodes: {e}")
        return nodes


class DocumentIndexingTask:
    """Class to embed documents from multiple sources into DuckDB index"""

    def __init__(self, index: DuckDBDocumentIndex, batch_size: int = 50, chunk_size: int = 1024, chunk_overlap: int = 50,
                 delay_between_batches: float = 2.0, delay_between_super_batches: float = 2.0,
                 create_embeddings: bool = True, max_retry_delay: int = 60,
                 base_retry_delay: float = 10.0, jitter: float = 1.0):
        self.index = index
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.delay_between_batches = delay_between_batches
        self.delay_between_super_batches = delay_between_super_batches
        self.create_embeddings = create_embeddings
        self.max_retry_delay = max_retry_delay
        self.base_retry_delay = base_retry_delay
        self.jitter = jitter

    def run(self, documents: List[Document]):
        """Main processing function with sequential processing"""
        print(f"Starting document processing (embeddings: {'enabled' if self.create_embeddings else 'disabled'})...")
        start_time = time.time()

        Settings.embed_model = OpenAIEmbedding()
        embed_model = Settings.embed_model

        removed_docs = self.remove_outdated_documents(documents, embed_model)
        if removed_docs:
            print(f"Removed nodes for {len(removed_docs)} documents with newer content")
        # Create nodes from documents
        nodes = self.create_nodes(documents)
        nodes = self.filter_existing_nodes(nodes, embed_model)

        if not nodes:
            print("No new nodes to process!")
            self._write_completion_file(0, 0, time.time() - start_time)
            return

        total_processed = 0
        total_written = 0

        if self.create_embeddings:
            # Create batches for embedding processing
            batches = [nodes[i:i + self.batch_size] for i in range(0, len(nodes), self.batch_size)]
            print(f"Created {len(batches)} batches for embedding processing")

            # Process each batch sequentially with embeddings
            with tqdm(total=len(nodes), desc="Processing") as pbar:
                for batch_idx, batch in enumerate(batches):
                    print(f"\nProcessing batch {batch_idx + 1}/{len(batches)}")

                    # Process batch using asyncio for embedding
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    try:
                        embedded_batch = loop.run_until_complete(self.embed_batch(embed_model, batch))
                        total_processed += len(embedded_batch)
                        processed_batch = embedded_batch

                    except Exception as e:
                        print(f"Error processing batch {batch_idx + 1}: {e}")
                        processed_batch = batch
                        total_processed += len(batch)
                    finally:
                        # Properly clean up pending tasks before closing loop
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()

                        # Wait for all tasks to be cancelled
                        if pending:
                            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

                        loop.close()

                    # Write batch to database
                    print(f"Writing {len(processed_batch)} nodes to database...")
                    written_count = self.write_nodes_to_db(processed_batch, embed_model)
                    total_written += written_count

                    # Update progress
                    pbar.update(len(processed_batch))

                    print(f"Batch {batch_idx + 1} complete: {len(processed_batch)} processed, {written_count} written")

                    # Add delay between batches to avoid overwhelming the API
                    if batch_idx < len(batches) - 1:  # Don't delay after the last batch
                        # Add jitter to avoid synchronized requests
                        delay = self.delay_between_super_batches + random.uniform(0, self.jitter)
                        print(f"Waiting {delay:.2f}s before next batch...")
                        time.sleep(delay)
        else:
            # When embeddings are disabled, process all nodes at once
            print(f"Processing all {len(nodes)} nodes at once (no embeddings)")
            total_processed = len(nodes)

            # Write all nodes to database in one go
            print(f"Writing {len(nodes)} nodes to database...")
            written_count = self.write_nodes_to_db(nodes, embed_model)
            total_written = written_count

            print(f"Processing complete: {total_processed} processed, {total_written} written")

        # Final stats
        total_time = time.time() - start_time

        print(f"\n=== Processing Complete ===")
        print(f"Total nodes processed: {total_processed}")
        print(f"Total nodes written: {total_written}")
        print(f"Embeddings created: {'Yes' if self.create_embeddings else 'No'}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average rate: {total_processed / total_time:.1f} nodes/second")

        self._write_completion_file(total_processed, total_written, total_time)


    def create_nodes(self, documents):
        """Create nodes from documents"""
        print("Creating nodes from documents...")
        splitter = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
        print(f"Created {len(nodes)} total nodes")
        return nodes

    def filter_existing_nodes(self, nodes, embed_model):
        """Remove nodes that are already embedded"""
        if not nodes:
            return []

        try:
            if self.create_embeddings:
                # Get all existing hashes
                existing_hashes = set(self.index.get_all_node_hashes())

                # Filter out nodes that already exist
                filtered_nodes = []
                for node in nodes:
                    if node.hash is None or node.hash not in existing_hashes:
                        filtered_nodes.append(node)

                print(f"Filtered out {len(nodes) - len(filtered_nodes)} existing nodes")
                return filtered_nodes
            else:
                # Use the pandas approach for filtering nodes without embeddings
                filtered_nodes = pandas_filter(nodes, self.index.db_path)
                return filtered_nodes
        except Exception as e:
            print(f"Error filtering nodes: {e}")
            return nodes

    def extract_wait_time(self, error_message):
        """Extract the wait time in seconds from Azure OpenAI error message"""
        # Look for patterns like "Please retry after 20 seconds"
        match = re.search(r'retry after (\d+) seconds', error_message)
        if match:
            return int(match.group(1))
        else:
            # Default wait time if we can't parse it
            return self.base_retry_delay

    async def embed_batch(self, embed_model, batch):
        """Embed a batch of nodes with rate limiting"""
        try:
            texts = [node.get_text() for node in batch]

            # Add delay before embedding to respect rate limits
            delay = self.delay_between_batches + random.uniform(0, self.jitter)
            print(f"Waiting {delay:.2f}s before embedding batch...")
            await asyncio.sleep(delay)

            # Function to embed with exact retry timing from API error
            async def _embed_with_retry():
                max_retries = 10
                retry_count = 0

                while True:
                    try:
                        return await embed_model.aembed_documents_batch(
                            texts,
                            batch_size=len(texts),
                            delay_between_batches=1.0  # Increased delay in the embedding call
                        )
                    except RateLimitError as e:
                        retry_count += 1
                        if retry_count > max_retries:
                            print(f"Maximum retries ({max_retries}) exceeded. Raising error.")
                            raise

                        error_message = str(e)
                        wait_time = self.extract_wait_time(error_message)

                        wait_time += 2

                        print(f"Rate limit hit. API requested wait time: {wait_time}s. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)

            embeddings = await _embed_with_retry()

            # Assign embeddings
            for node, embedding in zip(batch, embeddings):
                if embedding is not None:
                    node.embedding = embedding

            return batch
        except Exception as e:
            print(f"Error embedding batch: {e}")
            return batch

    def remove_outdated_documents(self, documents, embed_model):
        """Remove nodes for documents whose update_date is newer than what's in the database"""
        if not documents:
            return []

        print("Checking for documents with updated content...")

        # Group documents by ref_doc_id
        docs_by_id = {}
        for doc in documents:
            if hasattr(doc, 'doc_id') and doc.doc_id:
                # Use doc_id as ref_doc_id for database lookup
                docs_by_id[doc.doc_id] = doc

        # Get existing document info from database
        docs_to_check = list(docs_by_id.keys())
        docs_to_remove = []

        if not docs_to_check:
            return []

        # Create placeholders for SQL query
        placeholders = ', '.join(['?'] * len(docs_to_check))
        query = f"SELECT DISTINCT ref_doc_id, extra_data FROM documents WHERE ref_doc_id IN ({placeholders})"
        results = self.index.conn.execute(query, docs_to_check).fetchall()

        # Check each document for update_date
        for row in results:
            ref_doc_id, extra_data = row
            if ref_doc_id not in docs_by_id:
                continue

            # Get current document's update_date
            current_doc = docs_by_id[ref_doc_id]
            current_update_date = None
            if hasattr(current_doc,
                       'extra_info') and current_doc.extra_info and 'update_date' in current_doc.extra_info:
                current_update_date = current_doc.extra_info['update_date']

            # Get existing document's update_date
            existing_update_date = None
            existing_info = json.loads(extra_data) if extra_data else {}
            if existing_info and 'update_date' in existing_info:
                existing_update_date = existing_info['update_date']

            needs_update = False

            if current_update_date is None:
                continue

            if existing_update_date is None:
                needs_update = True
            else:
                if str(current_update_date) > str(existing_update_date):
                    needs_update = True

            if needs_update:
                docs_to_remove.append(ref_doc_id)

        if docs_to_remove:
            print(f"Removing {len(docs_to_remove)} outdated documents from database...")
            for ref_doc_id in docs_to_remove:
                print(f"Removing nodes for document {ref_doc_id}...")
                self.index.remove_document(ref_doc_id)

        else:
            print("No outdated documents found.")
        return docs_to_remove

    def write_nodes_to_db_pandas(self, nodes):
        """Write nodes to database using pandas and DuckDB when embeddings are disabled"""
        try:

            if not nodes:
                print("No nodes to write")
                return 0

            print(f"Converting {len(nodes)} nodes to DataFrame...")
            df = nodes_to_dataframe(nodes)

            print(f"Writing {len(df)} nodes to DuckDB using pandas...")
            num_written = save_nodes_to_duckdb(
                df,
                self.index.db_path,
                preserve_order=True,
                skip_existing=True
            )

            return num_written
        except Exception as e:
            print(f"Error writing nodes to database using pandas: {e}")
            return 0

    def _write_completion_file(self, total_processed, total_written, total_time):
        """Write a completion file with processing stats"""
        embedding_suffix = "with_embeddings" if self.create_embeddings else "without_embeddings"
        output_filename = f"output/document_processing_complete_{embedding_suffix}_{int(time.time())}.txt"

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        with open(output_filename, 'w') as f:
            if total_processed == 0 and total_written == 0:
                f.write(f"No new nodes to process - completed at {time.time()}")
            else:
                f.write(f"Processing completed at {time.time()}\n")
                f.write(f"Total nodes processed: {total_processed}\n")
                f.write(f"Total nodes written: {total_written}\n")
                f.write(f"Embeddings created: {'Yes' if self.create_embeddings else 'No'}\n")
                f.write(f"Total time: {total_time:.2f}s\n")

    def write_nodes_to_db(self, nodes, embed_model):
        """Write multiple nodes to database in a single bulk operation"""
        try:
            if self.create_embeddings:
                valid_nodes = [node for node in nodes if hasattr(node, 'embedding') and node.embedding is not None]
                if not valid_nodes:
                    print("No valid nodes with embeddings to write")
                    return 0

                print(f"Writing {len(valid_nodes)} nodes with embeddings in bulk operation...")
                self.index.update_nodes(valid_nodes, preserve_order=True, skip_existing=True)

                return len(valid_nodes)
            else:
                # Use pandas approach for nodes without embeddings
                return self.write_nodes_to_db_pandas(nodes)
        except Exception as e:
            print(f"Error writing to database: {e}")
            return 0
