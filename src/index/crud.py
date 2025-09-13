import duckdb
import json
from typing import List, Dict, Any, Optional, Set
from .models import DocumentNode

class CrudManager:
    """Handles CRUD operations for documents and nodes."""

    def __init__(self, conn: duckdb.DuckDBPyConnection, embed_model: object = None):
        self.conn = conn
        self.embed_model = embed_model

    def insert_nodes(self, nodes: List[BaseNode]) -> int:
        """Insert a list of nodes into the index."""
        if not nodes:
            return 0

        # Group nodes by ref_doc_id to process them together
        nodes_by_doc = {}
        for node in nodes:
            if node.ref_doc_id not in nodes_by_doc:
                nodes_by_doc[node.ref_doc_id] = []
            nodes_by_doc[node.ref_doc_id].append(node)

        total_inserted = 0
        for ref_doc_id, doc_nodes in nodes_by_doc.items():
            # Sort nodes by their order in the original document if possible
            sorted_nodes = sorted(doc_nodes, key=lambda n: n.metadata.get('order', 0))

            # Prepare data for insertion
            documents_data = []
            node_order_data = []
            for i, node in enumerate(sorted_nodes):
                extra_data = json.dumps(node.metadata)
                token_count = len(node.text.split())  # Simple token count

                embedding = None
                if self.embed_model:
                    embedding = self.embed_model.get_text_embedding(node.text)

                documents_data.append((
                    node.node_id,
                    node.ref_doc_id,
                    node.text,
                    extra_data,
                    embedding,
                    token_count,
                    node.hash
                ))
                node_order_data.append((node.ref_doc_id, node.node_id, i, token_count))

            # Bulk insert into documents and node_order tables
            self.conn.executemany("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)", documents_data)
            self.conn.executemany("INSERT INTO node_order VALUES (?, ?, ?, ?)", node_order_data)
            total_inserted += len(sorted_nodes)

        return total_inserted

    def get_nodes(self, node_ids: List[str]) -> List[DocumentNode]:
        """Retrieve nodes by their IDs."""
        if not node_ids:
            return []
        
        query = f"SELECT node_id, ref_doc_id, text, extra_data FROM documents WHERE node_id IN ({','.join(['?']*len(node_ids))})"
        results = self.conn.execute(query, node_ids).fetchall()
        
        nodes = []
        for row in results:
            node_id, ref_doc_id, text, extra_data = row
            metadata = json.loads(extra_data) if extra_data else {}
            node = DocumentNode(id_=node_id, text=text, metadata=metadata, ref_doc_id=ref_doc_id)
            nodes.append(node)
            
        return nodes

    def delete_ref_doc(self, ref_doc_id: str) -> None:
        """Delete all nodes associated with a ref_doc_id."""
        # First, delete from node_order, then from documents due to foreign key constraint
        self.conn.execute("DELETE FROM node_order WHERE ref_doc_id = ?", (ref_doc_id,))
        self.conn.execute("DELETE FROM documents WHERE ref_doc_id = ?", (ref_doc_id,))
