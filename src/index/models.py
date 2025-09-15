from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class DocumentNode(BaseModel):
    """Pydantic model for a document node, replacing TextNode."""
    node_id: str = Field(..., alias='id_')
    ref_doc_id: Optional[str] = None
    text: str
    metadata: Dict[str, Any] = {}

    class Config:
        allow_population_by_field_name = True
