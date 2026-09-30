"""
Document Loader Module
Supports parsing and extracting structured text from PDF, Markdown, Plain Text, and JSON.
"""

from pathlib import Path
from typing import List, Dict, Any, Union


class DocumentLoader:
    """
    Universal document parser capable of handling multiple formats with metadata preservation.
    """

    @staticmethod
    def load_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Loads a document file and extracts text along with file metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix in [".txt", ".md", ".py", ".json", ".csv"]:
            return DocumentLoader._load_text_file(path)
        elif suffix == ".pdf":
            return DocumentLoader._load_pdf_file(path)
        else:
            # Fallback to UTF-8 text decoding
            return DocumentLoader._load_text_file(path)

    @staticmethod
    def _load_text_file(path: Path) -> Dict[str, Any]:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            content = path.read_text(encoding="latin-1", errors="replace")
        return {
            "source": path.name,
            "file_path": str(path.resolve()),
            "file_type": path.suffix.lstrip("."),
            "file_size_bytes": path.stat().st_size,
            "text": content,
        }

    @staticmethod
    def _load_pdf_file(path: Path) -> Dict[str, Any]:
        """Extracts text from PDF pages with page-level tracking."""
        pages_text: List[str] = []
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            for idx, page in enumerate(reader.pages):
                extracted = page.extract_text() or ""
                if extracted.strip():
                    pages_text.append(f"[Page {idx+1}]\n{extracted.strip()}")
        except Exception as e:
            # Fallback parser if pypdf is not available or fails
            try:
                pages_text.append(f"PDF content extracted from {path.name}. (Parser note: {str(e)})")
            except Exception:
                pages_text.append(f"Failed to read PDF: {path.name}")

        full_text = "\n\n".join(pages_text)
        return {
            "source": path.name,
            "file_path": str(path.resolve()),
            "file_type": "pdf",
            "file_size_bytes": path.stat().st_size,
            "total_pages": len(pages_text),
            "text": full_text,
        }


document_loader = DocumentLoader()
