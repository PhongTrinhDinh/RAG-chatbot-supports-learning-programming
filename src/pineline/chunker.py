import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

import fitz
from bs4 import BeautifulSoup
import markdownify

# ==========================================================
#                       Data schema 
# ==========================================================

@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict
    
@dataclass
class ProcessedDocument:
    source: str
    source_type: str # "pdf"/"html"
    raw_content: str
    clean_text: str
    markdown_content: str
    chunks: List[Chunk] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
# ==========================================================
#                   Data cleaning
# ==========================================================    
    
class ContentCleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        
        text = text.replace("\xa0", " ")
        text = text.replace("\u200b", "")
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(f"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        
        return text.strip()
    
# =======================================================
#                  Markdown transformation
# =======================================================
    
class MarkdownTransformer:
    @staticmethod
    def html_to_markdown(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        
        # remove noise tag
        for tag in soup(["scripts", "style", "noscript", "iframe", "svg"]):
            tag.decompose()
            
        cleaned_html = str(soup)
        markdown = markdownify.markdownify(cleaned_html, heading_style="ATX")
        
        return ContentCleaner.clean_text(markdown)
    
    @staticmethod
    def text_to_markdown(text: str) -> str:
        markdown = markdownify.markdownify(text, heading_style="ATX")
        return ContentCleaner.clean_text(markdown)
    
# ===================================================================================================
#                            Chunking
# ===================================================================================================

class ChunkingStrategy:
    @staticmethod
    def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        if not text:
            return []

        # First, split by paragraphs (double newline)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # If a single paragraph is too long, split it using character-based chunking
            if len(paragraph) > chunk_size:
                # Finalize current chunk if it has content
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Split the long paragraph using original character-based approach
                para_chunks = ChunkingStrategy._split_text_character_based(paragraph, chunk_size, overlap)
                chunks.extend(para_chunks)
                continue

            # If adding this paragraph would exceed chunk_size and we have content
            if len(current_chunk) + len(paragraph) + 2 > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous chunk
                # Take last 'overlap' characters from current chunk as start of next
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If no chunks were created (shouldn't happen with valid input), fallback
        if not chunks:
            # Fallback to original character-based chunking
            chunks = []
            start = 0
            text_length = len(text)
            while start < text_length:
                end = min(start + chunk_size, text_length)
                chunk = text[start:end].strip()

                if chunk:
                    chunks.append(chunk)

                start += chunk_size - overlap

        return chunks

    @staticmethod
    def _split_text_character_based(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Original character-based chunking strategy as fallback"""
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start += chunk_size - overlap

        return chunks
    
# =========================================================================================
#                         Main orchestrator
# =========================================================================================

class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def process_pdf(self, file_path: str) -> ProcessedDocument:
        raw_text = self._extract_pdf_text(file_path)
        clean_text = ContentCleaner.clean_text(raw_text)
        markdown_content = MarkdownTransformer.text_to_markdown(clean_text)
        
        chunks = self._build_chunks(
            text=markdown_content,
            source=file_path,
            source_type="pdf"
        )
        
        return ProcessedDocument(
            source=file_path,
            source_type="pdf",
            raw_content=raw_text,
            clean_text=clean_text,
            markdown_content=markdown_content,
            chunks=chunks,
            metadata={
                "filename": Path(file_path).name
            }
        )
        
    def process_html(self, html: str, source: str = "scraped_html") -> ProcessedDocument:
        clean_html = ContentCleaner.clean_text(html)
        markdown_content = MarkdownTransformer.html_to_markdown(clean_html)
        clean_text = ContentCleaner.clean_text(markdown_content)
        
        chunks = self._build_chunks(
            text=markdown_content,
            source=source,
            source_type="html"
        )
        
        return ProcessedDocument(
            source=source,
            source_type="html",
            raw_content=html,
            clean_text=clean_text,
            markdown_content=markdown_content,
            chunks=chunks,
            metadata={}
        )
        
    def _extract_pdf_text(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        return text
    
    def _build_chunks(self, text: str, source: str, source_type: str) -> List[Chunk]:
        raw_chunks = ChunkingStrategy.split_text(
            text=text,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )
        
        return [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                metadata={
                    "source": source,
                    "source_type": source_type,
                    "chunk_index": idx
                }
            )
            for idx, chunk_text in enumerate(raw_chunks)
        ]
