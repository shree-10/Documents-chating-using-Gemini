import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from difflib import Differ
from docx import Document
from deep_translator import GoogleTranslator
from PIL import Image
import pytesseract
import io
from pdf2image import convert_from_path

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDFs, including OCR for scanned pages
def get_pdf_text_with_ocr(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if not page_text.strip():
                # Fallback to OCR for scanned pages
                images = convert_from_path(pdf, first_page=page_num + 1, last_page=page_num + 1, dpi=300)
                for image in images:
                    page_text = pytesseract.image_to_string(image)
            text += page_text + "\n"
    return text

# Function to extract text from Word files, including OCR for embedded images
def get_docx_text_with_ocr(docx_files):
    text = ""
    for docx in docx_files:
        document = Document(docx)
        # Extract text from paragraphs
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
        # Extract text from images
        for rel in document.part.rels.values():
            if "image" in rel.target_ref:
                image_data = rel.target_part.blob
                image = Image.open(io.BytesIO(image_data))
                text += pytesseract.image_to_string(image) + "\n"
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, say "Answer is not available in the context." Do not provide incorrect information.

    Context:
    {context}?

    Question: 
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

# Function to compare contents
def compare_texts(file_texts, file_names):
    st.subheader("Comparison Results:")

    for i in range(len(file_texts)):
        for j in range(i + 1, len(file_texts)):
            st.write(f"Comparing `{file_names[i]}` with `{file_names[j]}`:")
            differ = Differ()
            diff = list(differ.compare(file_texts[i].splitlines(), file_texts[j].splitlines()))
            for line in diff:
                if line.startswith("+"):
                    st.markdown(f"**Added in {file_names[j]}:** {line[2:]}")
                elif line.startswith("-"):
                    st.markdown(f"**Removed from {file_names[i]}:** {line[2:]}")

# Function to translate text
def translate_text(input_text, target_language="en"):
    try:
        translated = GoogleTranslator(source="auto", target=target_language).translate(input_text)
        return translated
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return ""

# Function to process uploaded files
def process_uploaded_files(uploaded_files):
    file_texts = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".pdf"):
            file_texts.append(get_pdf_text_with_ocr([uploaded_file]))
        elif uploaded_file.name.endswith(".docx"):
            file_texts.append(get_docx_text_with_ocr([uploaded_file]))
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            continue
    return "\n".join(file_texts)

# Main function
def main():
    st.set_page_config(page_title="Document Chat & Compare", page_icon="📊", layout="wide")
    st.header("🔗 Chat and Compare Documents using Gemini")

    # Tabs for different functionalities
    tabs = st.tabs(["Upload Files", "Compare Files", "Ask Questions", "OCR Translate Files"])

    with tabs[0]:
        st.subheader("📂 Upload Your Files")
        uploaded_files = st.file_uploader("Upload your PDF or Word Files", accept_multiple_files=True)
        extracted_text = ""
        if uploaded_files:
            if st.button("Process Files with OCR"):
                with st.spinner("Processing files with OCR..."):
                    extracted_text = process_uploaded_files(uploaded_files)
                    if extracted_text.strip():
                        st.text_area("Extracted Text", value=extracted_text, height=300)
                        st.success("File processing completed with OCR!")
                    else:
                        st.error("No text could be extracted. Please check the files.")

    with tabs[1]:
        st.subheader("🔄 Compare Files")
        if "file_texts" in locals() and "file_names" in locals() and file_texts and file_names:
            if st.button("Compare Documents"):
                compare_texts(file_texts, file_names)
        else:
            st.warning("Please upload and process files in the 'Upload Files' tab first.")

    with tabs[2]:
        st.subheader("🕵️ Ask Questions")
        user_question = st.text_input("Type your question below:")
        if user_question:
            user_input(user_question)

    with tabs[3]:
        st.subheader("🌍 OCR Translate Files")
        uploaded_translation_files = st.file_uploader("Upload Images or Scanned PDFs for OCR Translation", accept_multiple_files=True, type=["png", "jpg", "jpeg", "pdf"])
        if uploaded_translation_files:
            if st.button("Translate Text to English"):
                with st.spinner("Extracting and Translating..."):
                    extracted_text = process_uploaded_files(uploaded_translation_files)
                    translated_text = translate_text(extracted_text, target_language="en")
                    st.text_area("Translated Text", value=translated_text, height=300)
                    st.success("OCR and translation completed!")

if __name__ == "__main__":
    main()
