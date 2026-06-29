import PyPDF2
from docx import Document

def extract_text(file):
    file_name = file.name.lower()
    if file_name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file_name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text=""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    elif file_name.endswith(".docx"):
        doc = Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text +"\n"
        return text
    else:
        raise ValueError("Unsupported file type")


