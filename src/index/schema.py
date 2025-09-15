import duckdb

class SchemaManager:
    """Manages the database schema for the document index."""

    def __init__(self, conn: duckdb.DuckDBPyConnection, embedding_dim: int, use_embeddings: bool):
        self.conn = conn
        self.embedding_dim = embedding_dim
        self.use_embeddings = use_embeddings

    def create_tables(self):
        """Create the documents and node_order tables."""
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

        create_node_order_table_sql = """
        CREATE TABLE IF NOT EXISTS node_order (
            ref_doc_id VARCHAR NOT NULL,
            node_id VARCHAR NOT NULL,
            node_order INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            PRIMARY KEY (ref_doc_id, node_id),
            FOREIGN KEY (node_id) REFERENCES documents(node_id)
        );
        """
        self.conn.execute(create_node_order_table_sql)

        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_node_order_ref_doc_order
            ON node_order(ref_doc_id, node_order);
        """)

    def create_indexes(self):
        """Create FTS and vector similarity indexes."""
        try:
            result = self.conn.execute("""
                SELECT index_name 
                FROM duckdb_indexes() 
                WHERE table_name = 'documents' AND index_type = 'HNSW'
            """).fetchone()

            if self.use_embeddings and not result:
                self.conn.execute("""
                    CREATE INDEX documents_embedding_hnsw_idx
                    ON documents USING HNSW (embedding);
                """)

            # Create FTS index
            self.conn.execute("""
                PRAGMA create_fts_index(
                    'documents', 'node_id', 'text', overwrite=1
                );
            """)
        except Exception as e:
            print(f"Warning: Index creation failed: {e}")
