"""Thin shim: the painting engine lives in ~/AGi/capability/imaging (pip: aiva-imagegen)."""
from imagegen import MossClient, MossPaintError, paint_size
from imagegen.moss import PNG_SIGNATURE

__all__ = ['MossClient', 'MossPaintError', 'paint_size', 'PNG_SIGNATURE']
