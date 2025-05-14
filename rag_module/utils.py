import pypdf
import re

# file load functions
def pdf_to_text(filepath) -> str:
    """Extract text from a PDF file."""
    reader = pypdf.PdfReader(filepath)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()]) # join pages
    return text

def txt_to_text(filepath: str) -> str:
    """Read text from a .txt file."""
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()
    
def filter_citations(text):
    """
    Filter citations from a text.
    The pattern [number] and any preceding is removed.
    """
    filtered_text = re.sub(r'\s*\[\d+\]', '', text)
    return filtered_text