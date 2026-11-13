import os
from dotenv import load_dotenv
from .file_reader import extract_paragraphs
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


load_dotenv()

embedding = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", cache_dir=os.path.expanduser("~/.cache/fastembed"))
vectore_store = InMemoryVectorStore(embedding)

chunks = extract_paragraphs("pdfs/Penguins_ACL.pdf")
vectore_store.add_texts(texts=chunks)
print(f"indexed {len(chunks)} chunks")
