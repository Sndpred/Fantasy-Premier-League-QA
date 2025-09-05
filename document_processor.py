import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

#load environment variables
load_dotenv()

class DocumentProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, #size of each chunk
            chunk_overlap=200, #overlap between chunks
            length_function=len
        )
    def load_document(self, pdf_path):
        """Load PDF document"""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages")
        return documents
    
    def split_documents(self, documents):
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_embeddings(self, chunks):
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print("Embeddings created successfully")
        return vectorstore
    
#Test function
def test_document_processing():
    processor = DocumentProcessor()

    #find PDF file in documents folder
    documents_folder = "documents"
    pdf_files = [f for f in os.listdir(documents_folder) if f.endswith('.pdf')]

    if not pdf_files:
        print("Not found")
        return
        
    pdf_path = os.path.join(documents_folder, pdf_files[0])
    print(f"Using PDF:{pdf_files[0]}")

    #process document
    documents = processor.load_document(pdf_path)
    chunks = processor.split_documents(documents)

    # Show sample chunk
    print(f"\n Sample chunk (first 200 characters):")
    print("-" * 50)
    print(chunks[0].page_content[:200] + "...")
    print("-" * 50)
    print(f"Total chunks: {len(chunks)}")
        
    return processor, chunks

if __name__ == "__main__":
    test_document_processing()

