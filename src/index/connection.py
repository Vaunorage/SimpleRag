import duckdb
import os

class DuckDBConnection:
    """Handles DuckDB connection and extension loading."""

    def __init__(self, db_path: str, use_embeddings: bool = True):
        self.db_path = db_path
        self.use_embeddings = use_embeddings

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Establish a connection to the DuckDB database."""
        self._handle_wal_file()
        try:
            conn = duckdb.connect(self.db_path)
        except Exception as e:
            print(f"Error connecting to database: {e}")
            print("Removing corrupted database file and starting fresh...")
            try:
                os.remove(self.db_path)
            except OSError:
                pass
            conn = duckdb.connect(self.db_path)
        
        self._setup_extensions(conn)
        return conn

    def _handle_wal_file(self):
        """Handle the Write-Ahead Log (WAL) file if it exists."""
        wal_file = f"{self.db_path}.wal"
        if os.path.exists(wal_file):
            print(f"Warning: WAL file detected. Attempting recovery...")
            try:
                # Attempt a clean shutdown by connecting and closing
                conn = duckdb.connect(self.db_path)
                conn.close()
            except Exception as e:
                print(f"WAL recovery failed: {e}")
                print("Removing WAL file...")
                try:
                    os.remove(wal_file)
                except OSError:
                    pass

    def _setup_extensions(self, conn: duckdb.DuckDBPyConnection):
        """Install and load FTS and VSS extensions."""
        try:
            conn.execute("INSTALL fts;")
            conn.execute("LOAD fts;")
            if self.use_embeddings:
                conn.execute("INSTALL vss;")
                conn.execute("LOAD vss;")
                conn.execute("SET hnsw_enable_experimental_persistence = true;")
        except Exception as e:
            print(f"Warning: Extension setup failed: {e}")
