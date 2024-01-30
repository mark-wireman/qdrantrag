import os
import chromadb
from chromadb.utils import embedding_functions
from langchain.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama


CHROMA_DATA_PATH = "/home/app/data/"
EMBED_MODEL = "mistral"
COLLECTION_NAME = "demo_docs"

pdf_folder_path = "/mnt/docs"
ollama_ip_address = "192.168.50.36"
ollama_url = "http://"+ollama_ip_address+":11434"

loader = UnstructuredPDFLoader(pdf_folder_path,mode="elements",post_processors=[clean_extra_whitespace])
loader = DirectoryLoader(pdf_folder_path,silent_errors=True,show_progress=True)
documents = loader.load()

# split it into chunks
#text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
#docs = text_splitter.split_documents(documents)
docs = text_splitter.split_documents(documents)

print("\n\nSetting the Ollama embedding.\n\n")
oembed = OllamaEmbeddings(base_url=ollama_url, model="mistral")
# create the open-source embedding function
#embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#embedding_function = SentenceTransformerEmbeddings(oembed)

print("\n\nCreating the vector store and saving to Chroma.\n\n")
vectorstore = Chroma.from_documents(documents=docs, embedding=oembed, persist_directory=CHROMA_DATA_PATH)

# load docs into Chroma DB
#db = Chroma.from_documents(docs, embedding_function,persist_directory=CHROMA_DATA_PATH)
question="What is DevSecOps?"
print("\n")
print(question)
print("\n")

docs = vectorstore.similarity_search(question)
print("\n")
print(len(docs))
print("\n")

ollama = Ollama(base_url=ollama_url,model="mistral")

qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
print("\n")
print(qachain({"query": question}))
print("\n")