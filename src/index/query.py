import duckdb
from typing import List, Any, Optional
from .models import DocumentNode
import json

class QueryManager:
    """Handles vector, full-text, and hybrid search queries."""

    def __init__(self, conn: duckdb.DuckDBPyConnection, embed_model: object = None):
        self.conn = conn
        self.embed_model = embed_model

    def vector_search(self, query_embedding: List[float], top_k: int) -> List[DocumentNode]:
        """Perform a vector similarity search."""
        if not self.embed_model:
            return []

        query = """
        SELECT node_id, ref_doc_id, text, extra_data
        FROM documents
        ORDER BY list_cosine_similarity(embedding, ?) DESC
        LIMIT ?;
        """
        results = self.conn.execute(query, [query_embedding, top_k]).fetchall()
        return self._results_to_nodes(results)

    def fts_search(self, query_str: str, top_k: int) -> List[DocumentNode]:
        """Perform a full-text search."""
        query = """
        SELECT d.node_id, d.ref_doc_id, d.text, d.extra_data
        FROM documents d, fts_main_documents fts
        WHERE d.node_id = fts.node_id AND fts_main_documents MATCH ?
        LIMIT ?;
        """
        results = self.conn.execute(query, [query_str, top_k]).fetchall()
        return self._results_to_nodes(results)

    def hybrid_search(self, query_str: str, query_embedding: List[float], top_k: int, alpha: float) -> List[DocumentNode]:
        """Perform a hybrid search using both vector and FTS."""
        # This is a simplified hybrid search. A more advanced implementation might use Reciprocal Rank Fusion.
        vector_results = self.vector_search(query_embedding, top_k)
        fts_results = self.fts_search(query_str, top_k)

        # Combine and re-rank results (simple approach)
        combined_results = {node.node_id: node for node in vector_results}
        for node in fts_results:
            if node.node_id not in combined_results:
                combined_results[node.node_id] = node

        return list(combined_results.values())[:top_k]

    def _results_to_nodes(self, results: List[Any]) -> List[DocumentNode]:
        """Convert query results to a list of DocumentNode objects."""
        nodes = []
        for row in results:
            node_id, ref_doc_id, text, extra_data = row
            metadata = json.loads(extra_data) if extra_data else {}
            node = DocumentNode(id_=node_id, text=text, metadata=metadata, ref_doc_id=ref_doc_id)
            nodes.append(node)
        return nodes
