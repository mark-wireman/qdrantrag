#!/usr/bin/env python3
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
import chromadb
import os
import sys
import argparse
import time

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
chunk_size = CHUNKSIZE
chunk_overlap = CHUNKOVERLAP
hide_source = False

model = MODELNAME # os.environ.get("MODEL", "llama2-uncensored")
# For embeddings model, the example uses a sentence-transformers model
# https://www.sbert.net/docs/pretrained_models.html 
# "The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality."
target_source_chunks = 4 # int(os.environ.get('TARGET_SOURCE_CHUNKS',4))


def main():
    # Parse the command line arguments
    BASEURL = OLLAMAURL + ":" + OLLAMAPORTNO
    #embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    embeddings = OllamaEmbeddings(base_url=BASEURL, model=MODELNAME)

    print("Getting persist database.")
    db = Chroma(collection_name=COLLECTIONNAME,persist_directory=PERSISTDIR, embedding_function=embeddings)
    #db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    print("Getting the RetrievalQA.")
    llm = Ollama(base_url=BASEURL, model=MODELNAME, callbacks=callbacks)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=vectordb)
    #print("Getting retriever.")
    #retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs

    #print("Setting Ollama model.")
    #llm = Ollama(model=MODELNAME, callbacks=callbacks)
    
    #print("Getting the RetrievalQA.")
    #qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not hide_source)

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if hide_source else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

if __name__ == "__main__":
    main()