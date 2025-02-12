import PyPDF2

def process_pdf(file_path):
    chunks = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                # Treat each page as a single chunk
                # or split further if pages are large.
                chunks.append(text)
    return chunks
