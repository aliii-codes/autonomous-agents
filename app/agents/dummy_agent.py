import io
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings

load_dotenv()


index_name="pdf-rag-index"
embeddings = PineconeEmbeddings(model="multilingual-e5-large")


vector_store = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings
)

async def process_pdf(file):
    file_bytes = await file.read()
    
    
    pdf_stream = io.BytesIO(file_bytes)
    
    reader = PdfReader(pdf_stream)
    
    
    documents: List[Document] = []
    
    for page_num , page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            doc = Document(
                page_content=text,
                metadata={
                    "page" : page_num,
                    "source" : file.filename
                }
            )
            
            documents.append(doc)
            
            
    
    return documents



async def create_chunks(documents : List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=10,
        length_function=len
    )
    
    
    chunks = splitter.split_documents(documents=documents)
    
    
    print(f"bro the chunk is :{chunks[0]}")
    
    
    return chunks


async def store_in_pinecone(chunks: List[Document], namespace: str = None) -> PineconeVectorStore:
    store = await PineconeVectorStore.afrom_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=namespace
    )
    return store


async def search_documents(query: str, top_k: int = 4, namespace: str = None) -> List[Document]:
    """
    Queries Pinecone for the top_k most similar document chunks.
    """

    results = await vector_store.asimilarity_search(
        query=query,
        k=top_k,
        namespace=namespace
    )
    return results