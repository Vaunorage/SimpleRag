import shutil
from collections import defaultdict
from datetime import datetime
import duckdb
import json
import os
from typing import List, Dict, Any, Optional, Set
from llama_index.core.schema import BaseNode, TextNode

from src.utils import count_tokens


class DuckDBDocumentIndex:
    """
    A document index using DuckDB with full-text search (FTS) and vector similarity search (VSS) capabilities.
    Includes node ordering for retrieving adjacent chunks.
    """

    def __init__(self, db_path: str, embedding_dim: int = 1536, embed_model: object = None,
                 use_embeddings: bool = True) -> None:
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.embed_model = embed_model
        self.use_embeddings = use_embeddings

        self._setup_database_conn()
        self._setup_extensions()
        self._setup_tables()
        self._setup_indexes()

    def recreate_database(self, new_db_path: str = None, backup_old: bool = True) -> bool:
        """
        Recreate a corrupted DuckDB database with proper schema and data recovery.

        Args:
            new_db_path: Path for the new database (defaults to old_db_path + "_new")
            backup_old: Whether to backup the old database before replacing

        Returns:
            bool: True if successful, False otherwise
        """
        old_db_path = self.db_path

        # Set default new database path
        if new_db_path is None:
            base_name = old_db_path.replace('.db', '')
            new_db_path = f"{base_name}_new.db"

        print(f"Recreating database:")
        print(f"  Old: {old_db_path}")
        print(f"  New: {new_db_path}")
        print(f"  Embeddings: {self.use_embeddings} (dim: {self.embedding_dim})")

        # Close existing connection
        if hasattr(self, 'conn') and self.conn:
            try:
                self.conn.close()
            except:
                pass

        # Remove existing new database if it exists
        if os.path.exists(new_db_path):
            os.remove(new_db_path)
            print(f"Removed existing new database: {new_db_path}")

        # Remove WAL files that might cause issues
        for suffix in ['.wal', '.shm']:
            wal_file = old_db_path + suffix
            if os.path.exists(wal_file):
                os.remove(wal_file)
                print(f"Removed WAL file: {wal_file}")

        try:
            # Create new database connection
            new_conn = duckdb.connect(new_db_path)
            print("Created new database connection")

            # Install and load extensions
            print("Installing extensions...")
            new_conn.execute("INSTALL fts;")
            new_conn.execute("LOAD fts;")

            if self.use_embeddings:
                new_conn.execute("INSTALL vss;")
                new_conn.execute("LOAD vss;")
                new_conn.execute("SET hnsw_enable_experimental_persistence = true;")

            # Create tables
            print("Creating tables...")

            # Documents table
            create_documents_sql = f"""
            CREATE TABLE documents (
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
            new_conn.execute(create_documents_sql)

            # Node order table
            create_node_order_sql = """
            CREATE TABLE node_order (
                ref_doc_id VARCHAR NOT NULL,
                node_id VARCHAR NOT NULL,
                node_order INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                PRIMARY KEY (ref_doc_id, node_id),
                FOREIGN KEY (node_id) REFERENCES documents(node_id)
            );
            """
            new_conn.execute(create_node_order_sql)

            # Create index on node_order
            new_conn.execute("""
                CREATE INDEX idx_node_order_ref_doc_order
                ON node_order(ref_doc_id, node_order);
            """)

            print("Tables created successfully")

            # Try to recover data from old database
            data_recovered = False
            if os.path.exists(old_db_path):
                print("Attempting data recovery from old database...")
                try:
                    # Attach old database
                    new_conn.execute(f"ATTACH '{old_db_path}' AS old_db;")

                    # Check what tables exist in old database
                    tables_result = new_conn.execute("""
                        SELECT table_name 
                        FROM old_db.information_schema.tables 
                        WHERE table_schema = 'main'
                    """).fetchall()

                    available_tables = [row[0] for row in tables_result]
                    print(f"Available tables in old database: {available_tables}")

                    # Copy documents if table exists
                    if 'documents' in available_tables:
                        try:
                            # Get count first
                            count_result = new_conn.execute("SELECT COUNT(*) FROM old_db.documents").fetchone()
                            print(f"Found {count_result[0]} documents in old database")

                            if count_result[0] > 0:
                                # Copy data
                                new_conn.execute("""
                                    INSERT INTO documents 
                                    SELECT * FROM old_db.documents
                                """)
                                print(f"Copied {count_result[0]} documents")
                                data_recovered = True
                        except Exception as e:
                            print(f"Could not copy documents: {e}")

                    # Copy node_order if table exists
                    if 'node_order' in available_tables:
                        try:
                            count_result = new_conn.execute("SELECT COUNT(*) FROM old_db.node_order").fetchone()
                            print(f"Found {count_result[0]} node order entries in old database")

                            if count_result[0] > 0:
                                new_conn.execute("""
                                    INSERT INTO node_order 
                                    SELECT * FROM old_db.node_order
                                """)
                                print(f"Copied {count_result[0]} node order entries")
                                data_recovered = True
                        except Exception as e:
                            print(f"Could not copy node_order: {e}")

                    # Detach old database
                    new_conn.execute("DETACH old_db;")

                except Exception as e:
                    print(f"Data recovery failed: {e}")
                    print("Continuing with empty database...")

            # Create indexes
            print("Creating indexes...")

            # FTS index
            try:
                new_conn.execute("""
                    PRAGMA create_fts_index('documents', 'node_id', 'text');
                """)
                print("Created FTS index")
            except Exception as e:
                print(f"Warning: FTS index creation failed: {e}")

            # Vector index (only if we have embeddings and data)
            if self.use_embeddings and data_recovered:
                try:
                    # Check if we have any embeddings
                    embedding_count = new_conn.execute(
                        "SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL"
                    ).fetchone()

                    if embedding_count[0] > 0:
                        new_conn.execute("""
                            CREATE INDEX documents_embedding_hnsw_idx
                            ON documents USING HNSW (embedding);
                        """)
                        print(f"Created HNSW index for {embedding_count[0]} embeddings")
                    else:
                        print("No embeddings found, skipping HNSW index creation")
                except Exception as e:
                    print(f"Warning: HNSW index creation failed: {e}")

            # Force checkpoint
            new_conn.execute("CHECKPOINT;")
            print("Database checkpoint completed")

            # Get final stats
            doc_count = new_conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            order_count = new_conn.execute("SELECT COUNT(*) FROM node_order").fetchone()[0]

            print(f"\nNew database created successfully!")
            print(f"  Documents: {doc_count}")
            print(f"  Node orders: {order_count}")
            print(f"  Data recovered: {data_recovered}")

            # Close connection
            new_conn.close()

            # Backup old database and replace with new one
            if backup_old and os.path.exists(old_db_path):
                backup_path = f"{old_db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.move(old_db_path, backup_path)
                print(f"Backed up old database to: {backup_path}")

            # Move new database to old location
            shutil.move(new_db_path, old_db_path)
            print(f"New database moved to: {old_db_path}")

            # Reconnect to the new database
            self._setup_database_conn()
            self._setup_extensions()
            # Note: Tables and indexes are already created, so we don't need to call _setup_tables() and _setup_indexes()

            return True

        except Exception as e:
            print(f"Error recreating database: {e}")
            # Clean up on failure
            try:
                new_conn.close()
            except:
                pass
            if os.path.exists(new_db_path):
                os.remove(new_db_path)

            # Try to reconnect to original database
            try:
                self._setup_database_conn()
                self._setup_extensions()
            except:
                pass

            return False

    def _setup_database_conn(self):
        wal_file = f"{self.db_path}.wal"
        if os.path.exists(wal_file):
            print(f"Warning: WAL file detected. Attempting recovery...")
            try:
                temp_conn = duckdb.connect(":memory:")
                temp_conn.execute("INSTALL fts;")
                temp_conn.execute("LOAD fts;")
                if self.use_embeddings:
                    temp_conn.execute("INSTALL vss;")
                    temp_conn.execute("LOAD vss;")
                    temp_conn.execute("SET hnsw_enable_experimental_persistence = true;")
                temp_conn.close()

                self.conn = duckdb.connect(self.db_path)
            except Exception as e:
                print(f"WAL recovery failed: {e}")
                print("Removing WAL file and starting fresh...")
                try:
                    os.remove(wal_file)
                except:
                    pass
                self.conn = duckdb.connect(self.db_path)
        else:
            try:
                self.conn = duckdb.connect(self.db_path)
            except Exception as e:
                print(f"Error connecting to database: {e}")
                print("Removing corrupted database file and starting fresh...")
                try:
                    os.remove(self.db_path)
                except:
                    pass
                self.conn = duckdb.connect(self.db_path)

    def set_embed_model(self, embed_model):
        """Set or update the embedding model."""
        self.embed_model = embed_model

    def _setup_extensions(self):
        """Install and load FTS and VSS extensions."""
        try:
            self.conn.execute("INSTALL fts;")
            self.conn.execute("LOAD fts;")
            if self.use_embeddings:
                self.conn.execute("INSTALL vss;")
                self.conn.execute("LOAD vss;")
                self.conn.execute("SET hnsw_enable_experimental_persistence = true;")
        except Exception as e:
            print(f"Warning: Extension setup failed: {e}")

    def _setup_tables(self):
        """Create the documents and node_order tables."""
        # Main documents table
        create_documents_table_sql = f"""
        CREATE TABLE IF NOT EXISTS documents (
            node_id VARCHAR PRIMARY KEY,
            ref_doc_id VARCHAR NOT NULL,
            text VARCHAR NOT NULL,
            extra_data VARCHAR, -- JSON string
            embedding FLOAT[{self.embedding_dim}],
            token_count INTEGER,
            hash VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.conn.execute(create_documents_table_sql)

        # Node ordering table for adjacency tracking
        create_node_order_table_sql = """
                                      CREATE TABLE IF NOT EXISTS node_order
                                      (
                                          ref_doc_id
                                          VARCHAR
                                          NOT
                                          NULL,
                                          node_id
                                          VARCHAR
                                          NOT
                                          NULL,
                                          node_order
                                          INTEGER
                                          NOT
                                          NULL,
                                          token_count
                                          INTEGER
                                          NOT
                                          NULL,
                                          PRIMARY
                                          KEY
                                      (
                                          ref_doc_id,
                                          node_id
                                      ),
                                          FOREIGN KEY
                                      (
                                          node_id
                                      ) REFERENCES documents
                                      (
                                          node_id
                                      )
                                          ); \
                                      """
        self.conn.execute(create_node_order_table_sql)

        # Create index on node_order for efficient adjacency queries
        self.conn.execute("""
                          CREATE INDEX IF NOT EXISTS idx_node_order_ref_doc_order
                              ON node_order(ref_doc_id, node_order);
                          """)

    def _setup_indexes(self):
        """Create FTS and vector similarity indexes."""
        try:
            result = self.conn.execute("""
                                       SELECT COUNT(*)
                                       FROM information_schema.schemata
                                       WHERE schema_name = 'fts_main_documents'
                                       """).fetchone()

            if result[0] == 0:
                self.conn.execute("""
                    PRAGMA create_fts_index(
                        'documents', 'node_id', 'text'
                    );
                """)

            if self.use_embeddings:
                try:
                    count_result = self.conn.execute(
                        "SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL").fetchone()
                    if count_result[0] > 0:
                        self.conn.execute("""
                                          CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
                                              ON documents USING HNSW (embedding);
                                          """)
                    else:
                        print("Skipping HNSW index creation - no embeddings found")
                except Exception as e:
                    print(f"Warning: Vector index creation failed: {e}")

        except Exception as e:
            print(f"Warning: Index setup failed: {e}")

    def _node_to_dict(self, node: BaseNode, token_count: Optional[int] = None) -> Dict[str, Any]:
        """Convert BaseNode to dictionary format for database storage."""
        # Ensure text is properly encoded/decoded
        text = node.get_text() or ''
        try:
            # Try to ensure we have valid UTF-8 text
            # First try to decode if it's bytes
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            # If it's a string, ensure it's valid UTF-8
            elif isinstance(text, str):
                # Encode and then decode to replace invalid characters
                text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        except Exception as e:
            print(f"Warning: Text encoding issue for node {getattr(node, 'id_', 'unknown')}: {e}")
            # Fallback to replacing problematic characters
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
            else:
                text = str(text).encode('utf-8', errors='replace').decode('utf-8', errors='replace')

        # Handle extra_info encoding issues
        try:
            extra_data_json = json.dumps(node.extra_info) if node.extra_info else None
        except Exception as e:
            print(f"Warning: Extra data encoding issue for node {getattr(node, 'id_', 'unknown')}: {e}")
            # Try to sanitize the extra_info
            sanitized_extra_info = {}
            if node.extra_info:
                for k, v in node.extra_info.items():
                    try:
                        # Test if the value can be JSON serialized
                        json.dumps({k: v})
                        sanitized_extra_info[k] = v
                    except:
                        # If not, convert to string and sanitize
                        sanitized_extra_info[k] = str(v).encode('utf-8', errors='replace').decode('utf-8',
                                                                                                  errors='replace')
            extra_data_json = json.dumps(sanitized_extra_info) if sanitized_extra_info else None

        # Handle embedding - use existing or generate if embeddings are enabled and we have an embedding model
        embedding = None
        if self.use_embeddings:
            if hasattr(node, 'embedding') and node.embedding is not None:
                # Use existing embedding if available
                if isinstance(node.embedding, list):
                    embedding = node.embedding
                else:
                    embedding = node.embedding.tolist() if hasattr(node.embedding, 'tolist') else list(node.embedding)
            elif self.embed_model is not None and text:  # Use sanitized text
                # Generate embedding if we have an embedding model and the node has text
                try:
                    embedding = self.embed_model.get_text_embedding(text)
                    # Update the node's embedding attribute for future use
                    node.embedding = embedding
                except Exception as e:
                    print(f"Warning: Failed to generate embedding for node {getattr(node, 'id_', 'unknown')}: {e}")

        node_id = getattr(node, 'node_id', None) or getattr(node, 'id_', None) or str(id(node))

        ref_doc_id = ''
        if hasattr(node, 'ref_doc_id') and node.ref_doc_id:
            ref_doc_id = node.ref_doc_id
        elif node.extra_info and 'ref_doc_id' in node.extra_info:
            ref_doc_id = node.extra_info['ref_doc_id']
        elif hasattr(node, 'source_node') and node.source_node:
            ref_doc_id = getattr(node.source_node, 'ref_doc_id', '')

        # Calculate token count if not provided
        if token_count is None:
            token_count = count_tokens(text)  # Use sanitized text

        # Get the hash property if it exists (TextNode has this property)
        node_hash = None
        if hasattr(node, 'hash') and node.hash is not None:
            node_hash = node.hash

        return {
            'node_id': node_id,
            'ref_doc_id': ref_doc_id,
            'text': text,  # Use sanitized text
            'extra_data': extra_data_json,
            'embedding': embedding,
            'token_count': token_count,
            'hash': node_hash
        }

    def _dict_to_node(self, row: Dict[str, Any]) -> BaseNode:
        """Convert database row to BaseNode."""
        extra_info = json.loads(row['extra_data']) if row['extra_data'] else {}

        node = TextNode(
            id_=row['node_id'],
            text=row['text'],
            extra_info=extra_info
        )

        if row['ref_doc_id']:
            if 'ref_doc_id' not in node.extra_info:
                node.extra_info['ref_doc_id'] = row['ref_doc_id']

        # Add token count to extra_info
        if 'token_count' in row and row['token_count'] is not None:
            node.extra_info['token_count'] = row['token_count']

        if row.get('embedding'):
            node.embedding = row['embedding']

        return node

    def update_node(self, node: BaseNode, node_order: Optional[int] = None, token_count: Optional[int] = None,
                    skip_existing: bool = False):
        """Update or add a single document with optional ordering and token count.

        Args:
            node: The node to update or add
            node_order: Optional order for the node within its document
            token_count: Optional pre-calculated token count
            skip_existing: If True, skip updating if a document with the same hash already exists
        """
        # Calculate token count if not provided
        if token_count is None:
            token_count = count_tokens(node.text or '')

        node_dict = self._node_to_dict(node, token_count)

        # If skip_existing is True and the node has a hash, check if it already exists
        if skip_existing and node_dict['hash'] is not None:
            # Check if a document with this hash already exists
            check_sql = "SELECT node_id FROM documents WHERE hash = ?"
            result = self.conn.execute(check_sql, [node_dict['hash']]).fetchone()
            if result:
                # Document with this hash already exists, skip the update
                return

        # Insert/update document
        sql = """
        INSERT OR REPLACE INTO documents 
        (node_id, ref_doc_id, text, extra_data, embedding, token_count, hash, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """

        self.conn.execute(sql, [
            node_dict['node_id'],
            node_dict['ref_doc_id'],
            node_dict['text'],
            node_dict['extra_data'],
            node_dict['embedding'],
            node_dict['token_count'],
            node_dict['hash']
        ])

        # Insert/update node order if provided
        if node_order is not None and node_dict['ref_doc_id']:
            order_sql = """
            INSERT OR REPLACE INTO node_order (ref_doc_id, node_id, node_order, token_count)
            VALUES (?, ?, ?, ?)
            """
            self.conn.execute(order_sql, [
                node_dict['ref_doc_id'],
                node_dict['node_id'],
                node_order,
                node_dict['token_count']
            ])

        self._refresh_fts_index()
        if self.use_embeddings and node_dict['embedding'] is not None:
            self._refresh_vector_index()

    def update_nodes(self, nodes: List[BaseNode], preserve_order: bool = True, skip_existing: bool = False):
        """Update or add multiple documents, optionally preserving their order.

        Args:
            nodes: List of nodes to update or add
            preserve_order: Whether to preserve the order of nodes within a document
            skip_existing: If True, skip updating nodes that already exist in the database (based on hash)
        """
        if not nodes:
            return

        # Calculate token counts for all nodes
        node_dicts = []
        for node in nodes:
            token_count = count_tokens(node.text or '')
            node_dicts.append(self._node_to_dict(node, token_count))

        # If skip_existing is True, filter out nodes that already exist in the database
        if skip_existing:
            # Get all hashes that are not None
            hashes = [nd['hash'] for nd in node_dicts if nd['hash'] is not None]

            if hashes:
                # Prepare placeholders for the SQL query
                placeholders = ', '.join(['?'] * len(hashes))

                # Query for existing hashes
                sql = f"SELECT hash FROM documents WHERE hash IN ({placeholders})"
                results = self.conn.execute(sql, hashes).fetchall()

                # Get the set of existing hashes
                existing_hashes = {row[0] for row in results}

                # Filter out nodes with existing hashes
                node_dicts = [nd for nd in node_dicts if nd['hash'] is None or nd['hash'] not in existing_hashes]

                # If all nodes were filtered out, return early
                if not node_dicts:
                    return

        # Batch insert/update documents
        sql = """
        INSERT OR REPLACE INTO documents 
        (node_id, ref_doc_id, text, extra_data, embedding, token_count, hash, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """

        data = [
            [nd['node_id'], nd['ref_doc_id'], nd['text'], nd['extra_data'], nd['embedding'], nd['token_count'],
             nd['hash']]
            for nd in node_dicts
        ]

        self.conn.executemany(sql, data)

        # Insert node order information if preserve_order is True
        if preserve_order:
            # Group nodes by ref_doc_id to maintain order within each document
            doc_groups = {}
            for i, node_dict in enumerate(node_dicts):
                ref_doc_id = node_dict['ref_doc_id']
                if ref_doc_id:
                    if ref_doc_id not in doc_groups:
                        doc_groups[ref_doc_id] = []
                    doc_groups[ref_doc_id].append((node_dict['node_id'], i, node_dict['token_count']))

            # Insert order data
            order_sql = """
            INSERT OR REPLACE INTO node_order (ref_doc_id, node_id, node_order, token_count)
            VALUES (?, ?, ?, ?)
            """

            order_data = []
            for ref_doc_id, node_list in doc_groups.items():
                for node_id, order, token_count in node_list:
                    order_data.append([ref_doc_id, node_id, order, token_count])

            if order_data:
                self.conn.executemany(order_sql, order_data)

        self._refresh_fts_index()
        if self.use_embeddings:
            self._refresh_vector_index()

    def get_adjacent_nodes(self, node_id: str, token_size_window: int = 512,
                           include_current: bool = True) -> List[BaseNode]:
        """
        Get nodes adjacent to a given node_id based on a token window size.

        Args:
            node_id: The target node ID
            token_size_window: Total token budget for the window
            include_current: Whether to include the current node in results

        Returns:
            List of BaseNode objects in order, respecting the token window
        """
        # First, get the ref_doc_id, order, and token count of the target node
        target_info = self.conn.execute("""
                                        SELECT ref_doc_id, node_order, token_count
                                        FROM node_order
                                        WHERE node_id = ?
                                        """, [node_id]).fetchone()

        if not target_info:
            print(f"Node {node_id} not found in node_order table")
            return []

        ref_doc_id, target_order, target_token_count = target_info

        # Get all nodes for this document ordered by position
        all_nodes = self.conn.execute("""
                                      SELECT d.node_id,
                                             d.ref_doc_id,
                                             d.text,
                                             d.extra_data,
                                             d.embedding,
                                             no.node_order,
                                             no.token_count
                                      FROM documents d
                                               JOIN node_order no
                                      ON d.node_id = no.node_id
                                      WHERE no.ref_doc_id = ?
                                      ORDER BY no.node_order
                                      """, [ref_doc_id]).fetchall()

        # Find the target node index
        target_idx = None
        for i, row in enumerate(all_nodes):
            if row[5] == target_order:  # node_order column
                target_idx = i
                break

        if target_idx is None:
            return []

        # Calculate token window
        selected_nodes = []
        total_tokens = 0

        # Start with current node if requested
        if include_current:
            selected_nodes.append(all_nodes[target_idx])
            total_tokens += target_token_count

        before_idx = target_idx - 1
        after_idx = target_idx + 1

        while (before_idx >= 0 or after_idx < len(all_nodes)) and total_tokens < token_size_window:
            can_expand_before = before_idx >= 0
            can_expand_after = after_idx < len(all_nodes)

            if can_expand_before and can_expand_after:
                before_count = target_idx - before_idx
                after_count = after_idx - target_idx
                expand_before = before_count <= after_count
            elif can_expand_before:
                expand_before = True
            elif can_expand_after:
                expand_before = False
            else:
                break

            if expand_before:
                node_tokens = all_nodes[before_idx][6]  # token_count column
                if total_tokens + node_tokens <= token_size_window:
                    selected_nodes.insert(0, all_nodes[before_idx])
                    total_tokens += node_tokens
                    before_idx -= 1
                else:
                    before_idx = -1  # Stop expanding in this direction
            else:
                node_tokens = all_nodes[after_idx][6]  # token_count column
                if total_tokens + node_tokens <= token_size_window:
                    selected_nodes.append(all_nodes[after_idx])
                    total_tokens += node_tokens
                    after_idx += 1
                else:
                    after_idx = len(all_nodes)  # Stop expanding in this direction

        # Convert to BaseNode objects
        nodes = []
        for row in selected_nodes:
            row_dict = {
                'node_id': row[0],
                'ref_doc_id': row[1],
                'text': row[2],
                'extra_data': row[3],
                'embedding': row[4],
                'token_count': row[6]
            }
            node = self._dict_to_node(row_dict)
            # Add order information to extra_info
            node.extra_info['node_order'] = row[5]
            nodes.append(node)

        return nodes

    def get_document_nodes(self, ref_doc_id: str, ordered: bool = True) -> List[BaseNode]:
        """
        Get all nodes for a specific document.

        Args:
            ref_doc_id: The document ID
            ordered: Whether to return nodes in order

        Returns:
            List of BaseNode objects
        """
        if ordered:
            sql = """
                  SELECT d.node_id, d.ref_doc_id, d.text, d.extra_data, d.embedding, d.token_count, no.node_order
                  FROM documents d
                           JOIN node_order no
                  ON d.node_id = no.node_id
                  WHERE no.ref_doc_id = ?
                  ORDER BY no.node_order \
                  """
        else:
            sql = """
                  SELECT node_id, ref_doc_id, text, extra_data, embedding, token_count
                  FROM documents
                  WHERE ref_doc_id = ? \
                  """

        results = self.conn.execute(sql, [ref_doc_id]).fetchall()

        nodes = []
        for row in results:
            row_dict = {
                'node_id': row[0],
                'ref_doc_id': row[1],
                'text': row[2],
                'extra_data': row[3],
                'embedding': row[4],
                'token_count': row[5]
            }
            node = self._dict_to_node(row_dict)
            if ordered and len(row) > 6:
                node.extra_info['node_order'] = row[6]
            nodes.append(node)

        return nodes

    def remove_document(self, ref_doc_id: str):
        """Remove all nodes associated with a ref_doc_id."""
        self.conn.execute("DELETE FROM node_order WHERE ref_doc_id = ?", [ref_doc_id])

        self.conn.execute("""
                          DELETE
                          FROM documents
                          WHERE ref_doc_id = ?
                             OR extra_data LIKE ?
                          """, [ref_doc_id, f'%"ref_doc_id": "{ref_doc_id}"%'])

        self._refresh_fts_index()

    def _refresh_vector_index(self):
        """Refresh the vector index after document updates."""
        if not self.use_embeddings:
            return

        try:
            count_result = self.conn.execute("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL").fetchone()
            if count_result[0] > 0:
                try:
                    self.conn.execute("DROP INDEX IF EXISTS documents_embedding_hnsw_idx;")
                except:
                    pass

                self.conn.execute("""
                                  CREATE INDEX documents_embedding_hnsw_idx
                                      ON documents USING HNSW (embedding);
                                  """)
        except Exception as e:
            print(f"Warning: Vector index refresh failed: {e}")

    def _refresh_fts_index(self):
        """Refresh the FTS index after document updates."""
        try:
            self.conn.execute("PRAGMA drop_fts_index('documents');")
            self.conn.execute("""
                PRAGMA create_fts_index(
                    'documents', 'node_id', 'text', 'extra_data'
                );
            """)
        except Exception as e:
            print(f"Warning: FTS index refresh failed: {e}")

    def _apply_filters(self, base_query: str, filters: Dict[str, Any]) -> tuple[str, list]:
        """Apply filters to the base query with proper JSON field matching for DuckDB.
           Also validates that each filter value is valid according to get_filter_values.

        Returns:
            tuple: (modified_query, additional_params)
        Raises:
            ValueError: If any filter value is not valid for its field.
        """
        if not filters:
            return base_query, []

        # Validate filters
        valid_filter_values = self.get_filter_values()
        invalid_filters = set(filters).difference(set(valid_filter_values))
        if invalid_filters:
            raise ValueError(f"Invalid filter values: {invalid_filters}")

        filter_conditions = []
        filter_params = []

        for key, value in filters.items():
            # Handle list of values (OR condition within the field)
            if isinstance(value, list):
                if not value:  # Skip empty lists
                    continue

                value_conditions = []
                for v in value:
                    # Use DuckDB JSON functions for proper field matching
                    value_conditions.append(
                        "(JSON_EXTRACT_STRING(extra_data, '$.' || ?) = ? OR "
                        "JSON_EXTRACT_STRING(extra_data, '$.' || ?) LIKE ? OR "
                        "JSON_CONTAINS(JSON_EXTRACT(extra_data, '$.' || ?), JSON(?)))"
                    )
                    # Add parameters for each condition:
                    filter_params.extend([key, str(v)])  # Exact match
                    filter_params.extend([key, f'%{str(v)}%'])  # Pattern match
                    filter_params.extend([key, f'"{str(v)}"'])  # JSON contains

                if value_conditions:
                    filter_conditions.append(f"({' OR '.join(value_conditions)})")

            # Handle single values
            else:
                # Use DuckDB JSON functions for proper field matching
                filter_conditions.append(
                    "(JSON_EXTRACT_STRING(extra_data, '$.' || ?) = ? OR "
                    "JSON_EXTRACT_STRING(extra_data, '$.' || ?) LIKE ? OR "
                    "JSON_CONTAINS(JSON_EXTRACT(extra_data, '$.' || ?), JSON(?)))"
                )
                # Add parameters for the condition
                filter_params.extend([key, str(value)])  # Exact match
                filter_params.extend([key, f'%{str(value)}%'])  # Pattern match
                filter_params.extend([key, f'"{str(value)}"'])  # JSON contains

        if filter_conditions:
            filter_clause = " AND " + " AND ".join(filter_conditions)
            return base_query + filter_clause, filter_params

        return base_query, []

    def _apply_filters_fallback(self, base_query: str, filters: Dict[str, Any]) -> tuple[str, list]:
        """Fallback filter method using string matching for databases without JSON support.

        Returns:
            tuple: (modified_query, additional_params)
        """
        if not filters:
            return base_query, []

        filter_conditions = []
        filter_params = []

        for key, value in filters.items():
            if isinstance(value, list):
                if not value:  # Skip empty lists
                    continue

                value_conditions = []
                for v in value:
                    # Look for the key-value pair in the JSON string
                    # Pattern: "key": "value" or "key": ["value1", "value2"]
                    value_conditions.extend([
                        'extra_data LIKE ?',  # "key": "value"
                        'extra_data LIKE ?'  # "key": [..., "value", ...]
                    ])
                    filter_params.extend([
                        f'%"{key}": "{str(v)}"%',
                        f'%"{key}"%{str(v)}%'
                    ])

                if value_conditions:
                    filter_conditions.append(f"({' OR '.join(value_conditions)})")

            else:
                # Single value matching
                filter_conditions.append('(extra_data LIKE ? OR extra_data LIKE ?)')
                filter_params.extend([
                    f'%"{key}": "{str(value)}"%',
                    f'%"{key}"%{str(value)}%'
                ])

        if filter_conditions:
            filter_clause = " AND " + " AND ".join(filter_conditions)
            return base_query + filter_clause, filter_params

        return base_query, []

    def get_filter_values(self) -> Dict[str, Set[Any]]:

        # self._setup_database_conn()
        field_values = defaultdict(set)
        try:
            cursor = self.conn.execute(f"SELECT extra_data FROM documents WHERE extra_data IS NOT NULL")
            rows = cursor.fetchall()

            for row in rows:
                extra_data_str = row[0]

                try:
                    data = json.loads(extra_data_str)

                    for field_name, field_value in data.items():
                        if isinstance(field_value, str) and field_value.startswith('[') and field_value.endswith(']'):
                            try:
                                list_values = json.loads(field_value)
                                if isinstance(list_values, list):
                                    for item in list_values:
                                        field_values[field_name].add(str(item))
                                else:
                                    field_values[field_name].add(str(field_value))
                            except json.JSONDecodeError:
                                field_values[field_name].add(str(field_value))
                        elif isinstance(field_value, list):
                            for item in field_value:
                                field_values[field_name].add(str(item))
                        else:
                            field_values[field_name].add(str(field_value))

                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse JSON: {e}")
                    continue

        except Exception as e:
            print(f"Database error: {e}")
            return {}

        result = {}
        for field_name, values in field_values.items():
            result[field_name] = sorted(list(values))

        return result

    def search_by_bm25(self, query, filters: Dict[str, Any] = None, limit: int = 10, exclude_ids: List[str] = None) -> \
            List[BaseNode]:
        filters = filters or {}
        exclude_ids = exclude_ids or []

        if isinstance(query, str):
            queries = [query]
        elif isinstance(query, (list, tuple)):
            queries = list(query)
        else:
            raise ValueError("Query must be a string or list of strings")

        if not queries:
            return []

        try:
            query_conditions = []
            query_params = []

            for q in queries:
                query_conditions.append("fts_main_documents.match_bm25(node_id, ?) IS NOT NULL")
                query_params.append(q)

            score_expressions = []
            for i, q in enumerate(queries):
                score_expressions.append(f"COALESCE(fts_main_documents.match_bm25(node_id, ?), 0)")
                query_params.append(q)

            combined_score = "GREATEST(" + ", ".join(score_expressions) + ")"

            sql = f"""
            SELECT d.node_id, d.ref_doc_id, d.text, d.extra_data, d.embedding, d.token_count, 
                   {combined_score} AS score
            FROM documents d
            WHERE ({" OR ".join(query_conditions)})
            """

            if exclude_ids:
                placeholders = ",".join("?" * len(exclude_ids))
                sql += f" AND d.node_id NOT IN ({placeholders})"
                query_params.extend(exclude_ids)

            # Apply filters with the new method
            try:
                sql, filter_params = self._apply_filters(sql, filters)
                query_params.extend(filter_params)
            except Exception as e:
                print(f"JSON filtering failed, using fallback: {e}")
                sql, filter_params = self._apply_filters_fallback(sql, filters)
                query_params.extend(filter_params)

            sql += " ORDER BY score DESC LIMIT ?"
            query_params.append(limit)

            results = self.conn.execute(sql, query_params).fetchall()

            nodes = []
            for row in results:
                row_dict = {
                    'node_id': row[0],
                    'ref_doc_id': row[1],
                    'text': row[2],
                    'extra_data': row[3],
                    'embedding': row[4],
                    'token_count': row[5]
                }
                node = self._dict_to_node(row_dict)
                node.extra_info['bm25_score'] = row[6]
                nodes.append(node)

            return nodes

        except Exception as e:
            print(f"BM25 search failed: {e}")
            return []

    def search_by_like(self, query, filters: Dict[str, Any] = None, limit: int = 10, exclude_ids: List[str] = None) -> \
            List[BaseNode]:
        filters = filters or {}
        exclude_ids = exclude_ids or []

        if isinstance(query, str):
            queries = [query]
        elif isinstance(query, (list, tuple)):
            queries = list(query)
        else:
            raise ValueError("Query must be a string or list of strings")

        if not queries:
            return []

        try:
            like_patterns = [f'%{q.lower()}%' for q in queries]

            text_conditions = []
            extra_data_conditions = []
            ref_doc_conditions = []

            text_scores = []
            extra_data_scores = []
            ref_doc_scores = []

            params = []

            for pattern in like_patterns:
                text_conditions.append("LOWER(text) LIKE ?")
                extra_data_conditions.append("LOWER(extra_data) LIKE ?")
                ref_doc_conditions.append("LOWER(ref_doc_id) LIKE ?")

                text_scores.append("CASE WHEN LOWER(text) LIKE ? THEN 100 ELSE 0 END")
                extra_data_scores.append("CASE WHEN LOWER(extra_data) LIKE ? THEN 50 ELSE 0 END")
                ref_doc_scores.append("CASE WHEN LOWER(ref_doc_id) LIKE ? THEN 30 ELSE 0 END")

                params.extend([pattern] * 6)

            all_scores = text_scores + extra_data_scores + ref_doc_scores
            combined_score = "(" + " + ".join(all_scores) + ")"

            all_where_conditions = text_conditions + extra_data_conditions + ref_doc_conditions
            where_clause = "(" + " OR ".join(all_where_conditions) + ")"

            sql = f"""
            SELECT node_id, ref_doc_id, text, extra_data, embedding, token_count,
                   {combined_score} as relevance_score
            FROM documents
            WHERE {where_clause}
            """

            if exclude_ids:
                placeholders = ",".join("?" * len(exclude_ids))
                sql += f" AND node_id NOT IN ({placeholders})"
                params.extend(exclude_ids)

            # Apply filters with the new method
            try:
                sql, filter_params = self._apply_filters(sql, filters)
                params.extend(filter_params)
            except Exception as e:
                print(f"JSON filtering failed, using fallback: {e}")
                sql, filter_params = self._apply_filters_fallback(sql, filters)
                params.extend(filter_params)

            sql += " ORDER BY relevance_score DESC LIMIT ?"
            params.append(limit)

            results = self.conn.execute(sql, params).fetchall()

            nodes = []
            for row in results:
                row_dict = {
                    'node_id': row[0],
                    'ref_doc_id': row[1],
                    'text': row[2],
                    'extra_data': row[3],
                    'embedding': row[4],
                    'token_count': row[5]
                }
                node = self._dict_to_node(row_dict)
                node.extra_info['relevance_score'] = row[6]
                nodes.append(node)

            return nodes

        except Exception as e:
            print(f"LIKE search failed: {e}")
            return []

    def search_by_vector(self, query=None, query_embedding=None,
                         filters: Dict[str, Any] = None, limit: int = 10, exclude_ids: List[str] = None) -> List[
        BaseNode]:
        if not self.use_embeddings:
            print("Warning: Vector search requested but embeddings are disabled")
            return []

        filters = filters or {}
        exclude_ids = exclude_ids or []

        embeddings_to_use = []

        if query_embedding is not None:
            if isinstance(query_embedding[0], (int, float)):
                embeddings_to_use = [query_embedding]
            else:
                embeddings_to_use = query_embedding
        elif query is not None and self.embed_model is not None:
            if isinstance(query, str):
                queries = [query]
            elif isinstance(query, (list, tuple)):
                queries = list(query)
            else:
                raise ValueError("Query must be a string or list of strings")

            for q in queries:
                embedding = self.embed_model.get_text_embedding(q)
                embeddings_to_use.append(embedding)
        else:
            raise ValueError("Either query_embedding must be provided, or both query and embed_model must be available")

        if not embeddings_to_use:
            return []

        for emb in embeddings_to_use:
            if len(emb) != self.embedding_dim:
                raise ValueError(f"Query embedding dimension {len(emb)} doesn't match expected {self.embedding_dim}")

        try:
            distance_expressions = []

            for emb in embeddings_to_use:
                embedding_str = f"[{','.join(map(str, emb))}]::FLOAT[{self.embedding_dim}]"
                distance_expressions.append(f"array_distance(embedding, {embedding_str})")

            combined_distance = "LEAST(" + ", ".join(distance_expressions) + ")"

            sql = f"""
            SELECT node_id, ref_doc_id, text, extra_data, embedding, token_count,
                   {combined_distance} AS distance
            FROM documents
            WHERE embedding IS NOT NULL
            """

            params = []
            if exclude_ids:
                placeholders = ",".join("?" * len(exclude_ids))
                sql += f" AND node_id NOT IN ({placeholders})"
                params.extend(exclude_ids)

            # Apply filters with the new method
            try:
                sql, filter_params = self._apply_filters(sql, filters)
                params.extend(filter_params)
            except Exception as e:
                print(f"JSON filtering failed, using fallback: {e}")
                sql, filter_params = self._apply_filters_fallback(sql, filters)
                params.extend(filter_params)

            sql = f"WITH s1 as ({sql}) SELECT * FROM s1 ORDER BY distance ASC LIMIT ?"
            params.append(limit)

            results = self.conn.execute(sql, params).fetchall()

            nodes = []
            for row in results:
                row_dict = {
                    'node_id': row[0],
                    'ref_doc_id': row[1],
                    'text': row[2],
                    'extra_data': row[3],
                    'embedding': row[4],
                    'token_count': row[5]
                }
                node = self._dict_to_node(row_dict)
                node.extra_info['vector_distance'] = row[6]
                nodes.append(node)

            return nodes

        except Exception as e:
            print(f"Vector search failed: {e}")
            return []

    def search_hybrid(self, query=None, query_embedding=None,
                      use_bm25=True, use_like=True, use_vector=True,
                      filters: Dict[str, Any] = None, limit: int = 10,
                      exclude_ids: List[str] = None) -> List[BaseNode]:
        """
        Unified search function that combines BM25, LIKE, and vector search in a single SQL query.
        """
        filters = filters or {}
        exclude_ids = exclude_ids or []

        # Process query inputs
        queries = []
        if query is not None:
            if isinstance(query, str):
                queries = [query]
            elif isinstance(query, (list, tuple)):
                queries = list(query)
            else:
                raise ValueError("Query must be a string or list of strings")

        # Process embedding inputs
        embeddings_to_use = []
        if use_vector:
            if query_embedding is not None:
                if isinstance(query_embedding[0], (int, float)):
                    embeddings_to_use = [query_embedding]
                else:
                    embeddings_to_use = query_embedding
            elif queries and self.embed_model is not None:
                for q in queries:
                    embedding = self.embed_model.get_text_embedding(q)
                    embeddings_to_use.append(embedding)
            elif use_vector:
                raise ValueError("Vector search enabled but no embeddings available")

        # Validate embeddings
        for emb in embeddings_to_use:
            if len(emb) != self.embedding_dim:
                raise ValueError(f"Query embedding dimension {len(emb)} doesn't match expected {self.embedding_dim}")

        if not queries and not embeddings_to_use:
            return []

        try:
            # Build score components
            score_components = []
            params = []
            where_conditions = []

            # BM25 scoring
            if use_bm25 and queries:
                bm25_expressions = []
                bm25_conditions = []

                for q in queries:
                    bm25_conditions.append("fts_main_documents.match_bm25(node_id, ?) IS NOT NULL")
                    bm25_expressions.append("COALESCE(fts_main_documents.match_bm25(node_id, ?), 0)")
                    params.extend([q, q])

                if bm25_conditions:
                    where_conditions.append("(" + " OR ".join(bm25_conditions) + ")")
                    bm25_score = "GREATEST(" + ", ".join(bm25_expressions) + ")"
                    score_components.append(f"({bm25_score}) * 1.0")  # BM25 weight = 1.0

            # LIKE scoring
            if use_like and queries:
                like_patterns = [f'%{q.lower()}%' for q in queries]

                text_conditions = []
                extra_data_conditions = []
                ref_doc_conditions = []
                text_scores = []
                extra_data_scores = []
                ref_doc_scores = []

                for pattern in like_patterns:
                    text_conditions.append("LOWER(text) LIKE ?")
                    extra_data_conditions.append("LOWER(extra_data) LIKE ?")
                    ref_doc_conditions.append("LOWER(ref_doc_id) LIKE ?")

                    text_scores.append("CASE WHEN LOWER(text) LIKE ? THEN 100 ELSE 0 END")
                    extra_data_scores.append("CASE WHEN LOWER(extra_data) LIKE ? THEN 50 ELSE 0 END")
                    ref_doc_scores.append("CASE WHEN LOWER(ref_doc_id) LIKE ? THEN 30 ELSE 0 END")

                    params.extend([pattern] * 6)

                all_like_conditions = text_conditions + extra_data_conditions + ref_doc_conditions
                if all_like_conditions:
                    where_conditions.append("(" + " OR ".join(all_like_conditions) + ")")

                    all_scores = text_scores + extra_data_scores + ref_doc_scores
                    like_score = "(" + " + ".join(all_scores) + ")"
                    score_components.append(f"({like_score}) * 0.01")  # LIKE weight = 0.01 to normalize

            # Vector scoring (converted to similarity, higher is better)
            if use_vector and embeddings_to_use:
                distance_expressions = []

                for emb in embeddings_to_use:
                    embedding_str = f"[{','.join(map(str, emb))}]::FLOAT[{self.embedding_dim}]"
                    distance_expressions.append(f"array_distance(embedding, {embedding_str})")

                if distance_expressions:
                    where_conditions.append("embedding IS NOT NULL")
                    combined_distance = "LEAST(" + ", ".join(distance_expressions) + ")"
                    # Convert distance to similarity: higher score = better match
                    vector_score = f"(1.0 / (1.0 + {combined_distance}))"
                    score_components.append(f"({vector_score}) * 100.0")  # Vector weight = 100.0

            # Combine all scores
            if not score_components:
                raise ValueError("At least one search method must be enabled")

            combined_score = " + ".join(score_components)

            # Build WHERE clause
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " OR ".join([f"({cond})" for cond in where_conditions])

            # Build main SQL
            sql = f"""
            SELECT node_id, ref_doc_id, text, extra_data, embedding, token_count,
                   {combined_score} AS unified_score
            FROM documents
            {where_clause}
            """

            # Add exclusion filter
            if exclude_ids:
                conjunction = "WHERE" if not where_clause else "AND"
                placeholders = ",".join("?" * len(exclude_ids))
                sql += f" {conjunction} node_id NOT IN ({placeholders})"
                params.extend(exclude_ids)

            # Apply additional filters with the new method
            try:
                sql, filter_params = self._apply_filters(sql, filters)
                params.extend(filter_params)
            except Exception as e:
                print(f"JSON filtering failed, using fallback: {e}")
                sql, filter_params = self._apply_filters_fallback(sql, filters)
                params.extend(filter_params)

            # Order and limit
            sql += " ORDER BY unified_score DESC LIMIT ?"
            params.append(limit)

            # Execute query
            results = self.conn.execute(sql, params).fetchall()

            # Convert results to nodes
            nodes = []
            
            # Get the maximum score for percentage calculation
            max_score = 1.0
            if results:
                max_score = max(row[6] for row in results) or 1.0
                
            for row in results:
                row_dict = {
                    'node_id': row[0],
                    'ref_doc_id': row[1],
                    'text': row[2],
                    'extra_data': row[3],
                    'embedding': row[4],
                    'token_count': row[5]
                }
                node = self._dict_to_node(row_dict)
                raw_score = row[6]
                node.extra_info['unified_score'] = raw_score
                
                # Calculate percentage score (0-100%)
                percentage = min(100.0, (raw_score / max_score) * 100.0)
                node.extra_info['score_percentage'] = round(percentage, 1)
                
                nodes.append(node)

            return nodes

        except Exception as e:
            print(f"Unified search failed: {e}")
            return []

    def search(self, query, filters: Dict[str, Any] = None, limit: int = 10,
               search_type: str = "hybrid", query_embedding: Optional[List[float]] = None,
               exclude_ids: List[str] = None) -> List[BaseNode]:
        # Adjust search_type if embeddings are disabled and vector search was requested
        if not self.use_embeddings and search_type == "vector":
            print("Warning: Vector search requested but embeddings are disabled, falling back to BM25")
            search_type = "bm25"

        filters = filters or {}
        exclude_ids = exclude_ids or []

        if search_type == "bm25":
            return self.search_by_bm25(query, filters, limit, exclude_ids)
        elif search_type == "vector":
            return self.search_by_vector(query=query, query_embedding=query_embedding,
                                         filters=filters, limit=limit, exclude_ids=exclude_ids)
        elif search_type == "like":
            return self.search_by_like(query, filters, limit, exclude_ids)
        elif search_type == "hybrid":
            return self.search_hybrid(query=query, filters=filters, limit=limit, query_embedding=query_embedding,
                                      exclude_ids=exclude_ids)
        else:
            raise ValueError("search_type must be 'bm25', 'vector', 'like', 'hybrid', or 'comprehensive'")

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_all_node_hashes(self) -> List[str]:
        """Return all node hashes that have been saved in the database.

        Returns:
            List[str]: A list of all node hashes stored in the database
        """
        try:
            sql = "SELECT hash FROM documents WHERE hash IS NOT NULL"
            results = self.conn.execute(sql).fetchall()

            # Extract hashes from results (each result is a tuple with one element)
            hashes = [row[0] for row in results]
            return hashes

        except Exception as e:
            print(f"Error retrieving node hashes: {e}")
            return []
