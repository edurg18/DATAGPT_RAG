import PyPDF2

def read_pdf(pdf_path):

    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text


def create_chunks(text, size_chunk):

    chunk_text = []

    for i in range(0, len(text), size_chunk):
        chunk_text.append(text[i:i+size_chunk])

    return chunk_text


