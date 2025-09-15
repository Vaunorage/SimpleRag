"""SimpleRAG: A comprehensive document indexing and retrieval system."""

from .core import (
    SimpleRAG,
    load_documents_from_folder,
    lms_vlm_options,
    lms_olmocr_vlm_options,
    ollama_vlm_options,
    watsonx_vlm_options,
)

__all__ = [
    "SimpleRAG",
    "load_documents_from_folder",
    "lms_vlm_options",
    "lms_olmocr_vlm_options",
    "ollama_vlm_options",
    "watsonx_vlm_options",
]
