# import libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


#Load data from pdf
def pdf_loader(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded document from {file_path}")
    return documents

# Create chunks 
def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)
    print(f"{len(chunks)} chunks created")

    return chunks

# Create vectorStore
def create_vector_store(chunks, persist_directory="db/chroma_db"):

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    print(f"Vector store created at {persist_directory}")

    return vector_store

# Run all pipline
def main():

    file_path = "D:\projects\"
    persist_directory = "vector_store"

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    if os.path.exists(persist_directory):

        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )

        return vector_store

    documents = pdf_loader(file_path)

    chunks = chunk_documents(documents)

    vector_store = create_vector_store(chunks, persist_directory)

    return vector_store


if __name__ == "__main__":
    main()
