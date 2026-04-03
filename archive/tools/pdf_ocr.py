#!/usr/bin/env python3
"""Extract text from scanned image PDFs using OCR (Tesseract).

Usage:
    python3 tools/pdf_ocr.py <pdf_path> [--output <txt_path>] [--pages 1-10] [--dpi 300]

Examples:
    # OCR entire PDF to stdout
    python3 tools/pdf_ocr.py book.pdf

    # OCR pages 1-50, save to file
    python3 tools/pdf_ocr.py book.pdf --output /tmp/book.txt --pages 1-50

    # Higher DPI for small text (slower but more accurate)
    python3 tools/pdf_ocr.py book.pdf --dpi 400

Requirements:
    brew install tesseract poppler
    pip3 install pytesseract pdf2image pillow
"""

import argparse
import sys
from pathlib import Path

try:
    from pdf2image import convert_from_path
    import pytesseract
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: brew install tesseract poppler && pip3 install pytesseract pdf2image")
    sys.exit(1)


def ocr_pdf(pdf_path: str, output_path: str | None = None,
            pages: str | None = None, dpi: int = 300) -> str:
    """Extract text from a scanned PDF using OCR.

    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to write extracted text
        pages: Page range like "1-50" or "5" or "10-20"
        dpi: Resolution for rendering (higher = more accurate, slower)

    Returns:
        Extracted text as a string
    """
    pdf = Path(pdf_path)
    if not pdf.exists():
        print(f"Error: {pdf_path} not found", file=sys.stderr)
        sys.exit(1)

    # Parse page range
    first_page, last_page = None, None
    if pages:
        if "-" in pages:
            parts = pages.split("-")
            first_page, last_page = int(parts[0]), int(parts[1])
        else:
            first_page = last_page = int(pages)

    print(f"Converting PDF pages to images (DPI={dpi})...", file=sys.stderr)

    kwargs = {"dpi": dpi, "fmt": "png"}
    if first_page:
        kwargs["first_page"] = first_page
    if last_page:
        kwargs["last_page"] = last_page

    images = convert_from_path(str(pdf), **kwargs)
    total = len(images)
    print(f"Got {total} pages. Running OCR...", file=sys.stderr)

    all_text = []
    for i, img in enumerate(images, start=first_page or 1):
        text = pytesseract.image_to_string(img)
        all_text.append(f"--- Page {i} ---\n{text}")
        if (i - (first_page or 1) + 1) % 10 == 0:
            print(f"  OCR'd {i - (first_page or 1) + 1}/{total} pages...", file=sys.stderr)

    result = "\n\n".join(all_text)

    if output_path:
        Path(output_path).write_text(result, encoding="utf-8")
        lines = result.count("\n") + 1
        print(f"Wrote {lines} lines to {output_path}", file=sys.stderr)

    return result


def main():
    parser = argparse.ArgumentParser(description="OCR scanned image PDFs to text")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--output", "-o", help="Output text file path")
    parser.add_argument("--pages", "-p", help="Page range (e.g., '1-50', '10-20', '5')")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI (default: 300)")
    args = parser.parse_args()

    text = ocr_pdf(args.pdf, args.output, args.pages, args.dpi)
    if not args.output:
        print(text)


if __name__ == "__main__":
    main()
