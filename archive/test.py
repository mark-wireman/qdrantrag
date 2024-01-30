#from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
#from unstructured.partition.auto import partition

directory = '/mnt/docs'

def load_docs(directory):
  loader = DirectoryLoader(directory,silent_errors=True,show_progress=True)
  documents = loader.load()
  doc_sources = [doc.metadata['source']  for doc in documents]
  print("\n\nDocument sources:\n\t")
  print(doc_sources)
  print("\n")
  return documents

documents = load_docs(directory)
print("\n\nNumber of loaded documents:\n\t")
print(len(documents))
print("\n")