#!/usr/bin/env python3
import os
import getopt, sys
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
import nltk
import argparse
import traceback
import logging

nltk.download('punkt')

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from chromadb.config import Settings


# Define the folder for storing database
MODELNAME = sys.argv[1] #"mistral"
CHUNKSIZE = sys.argv[2] #1500
CHUNKOVERLAP = sys.argv[3] #50
OLLAMAURL = sys.argv[4] #"http://localhost"
OLLAMAPORTNO = sys.argv[5] #"11434"
PERSISTDIR = sys.argv[6] #"/mnt/data"
SOURCEDIR = sys.argv[7] #"/mnt/docs"
COLLECTIONNAME = "v_db"

persist_directory = PERSISTDIR
source_directory = SOURCEDIR
embeddings_model_name = MODELNAME
chunk_size = int(CHUNKSIZE)
chunk_overlap = int(CHUNKOVERLAP)
no_of_docs = 0

print(("Ollama Server URL is %s\n") % (OLLAMAURL))


# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        persist_directory=PERSISTDIR,
        anonymized_telemetry=False
)

# Load environment variables
persist_directory = PERSISTDIR #'/mnt/data'
source_directory = SOURCEDIR #'/mnt/docs'
chunk_size = CHUNKSIZE #1500
chunk_overlap = CHUNKOVERLAP #50

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".xls": (UnstructuredExcelLoader, {}),
    # Add more mappings for other file extensions and loaders as needed
}

def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    no_of_docs = len(filtered_files)

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    print(documents)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {no_of_docs} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=20,length_function = len)
    #pages = text_splitter.split_text(documents)
    #texts = text_splitter.create_documents(pages)
    #texts = text_splitter.create_documents([documents['text']])
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts


def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def main():
    # Create embeddings
    #embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    BASEURL = OLLAMAURL + ":" + OLLAMAPORTNO
    oembed = OllamaEmbeddings(base_url=BASEURL, model=MODELNAME)

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding=oembed, collection_name=COLLECTIONNAME)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
        db.persist()
        db = None
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embedding=oembed, persist_directory=persist_directory, collection_name=COLLECTIONNAME)
        db.persist()
        db = None
   


if __name__ == "__main__":
    #try:
    main()
    #except Exception as e:
    print('An exception occurred: {}'.format(e))
    #finally:
    #db = None
    print('Done!')

