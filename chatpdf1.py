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

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from Word documents
def get_docx_text(docx_files):
    text = ""
    for docx in docx_files:
        document = Document(docx)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
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

# Main function
# Main function
def main():
    st.set_page_config(page_title="Document Chat & Compare", page_icon="📊", layout="wide")
    st.header("🔗 Chat and Compare Documents using Gemini")

    # Tabs for different functionalities
    tabs = st.tabs(["Upload Files", "Compare Files", "Ask Questions"])

    with tabs[0]:
        st.subheader("📂 Upload Your Files")
        uploaded_files = st.file_uploader("Upload your PDF or Word Files", accept_multiple_files=True)
        extracted_text = ""
        if uploaded_files:
            file_texts = []
            file_names = []
            for uploaded_file in uploaded_files:
                if uploaded_file.name.endswith(".pdf"):
                    file_texts.append(get_pdf_text([uploaded_file]))
                elif uploaded_file.name.endswith(".docx"):
                    file_texts.append(get_docx_text([uploaded_file]))
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue
                file_names.append(uploaded_file.name)

            if st.button("Process Files"):
                with st.spinner("Processing..."):
                    extracted_text = "\n".join(file_texts)
                    text_chunks = get_text_chunks(extracted_text)
                    get_vector_store(text_chunks)
                    st.success("Files processed and vector store created!")

        if extracted_text:
            st.subheader("Extracted Text")
            st.text_area("File Text", value=extracted_text, height=300)

            # JavaScript to copy text
            copy_button = """
            <button onclick="copyToClipboard()">Copy Text</button>
            <script>
            function copyToClipboard() {
                const textArea = document.querySelector('textarea');
                textArea.select();
                document.execCommand('copy');
                alert('Text copied to clipboard!');
            }
            </script>
            """
            st.markdown(copy_button, unsafe_allow_html=True)

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

if __name__ == "__main__":
    main()
