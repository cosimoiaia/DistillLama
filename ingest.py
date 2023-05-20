"""
Mainly based on:
https://python.langchain.com/en/latest/modules/agents/agent_executors/examples/agent_vectorstore.html

"""
import os
import glob
from typing import List, Any, Dict
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS

from pydantic import BaseModel
from langchain.embeddings.base import Embeddings

persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
model_path = os.environ.get('MODEL_PATH')

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {".txt": (TextLoader, {"encoding": "utf8"})}

load_dotenv()


class LLamaEmbeddings(Embeddings, BaseModel):
    model: Any
    model_path: str = model_path

    def __init__(self, **kwargs: Any):
        """Initialize the Llama model."""
        super().__init__(**kwargs)
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ValueError(
                "Could not import llama_cpp python package. "
                "Please install it with `pip install llama_cpp`."
            ) from exc

        self.model = Llama(model_path=self.model_path, embedding=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        #print(f"Embedding documents: {texts}")
        embeddings = [self.model.create_embedding(t)['data'][0]['embedding'] for t in texts]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        #print("Embedding query")
        embeddings = self.model.create_embedding(text)['data'][0]['embedding']
        return embeddings


def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from the source directory
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )

    from tqdm import tqdm

    progress_bar = tqdm(total=len(all_files), desc="Processing files", unit="file")

    results = []
    for file_path in all_files:
        results.append(load_single_document(file_path))
        progress_bar.update(1)

    progress_bar.close()

    return results

chunk_size = 500
chunk_overlap = 50

def ingest(source_directory:str = source_directory):
    print(f"Ingesting documents from {source_directory}")
    documents = load_documents(source_directory)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {source_directory}")
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} characters each)")

    # Create and store locally vectorstore
    db = Chroma.from_documents(texts, LLamaEmbeddings(model_path=model_path), persist_directory=persist_directory,
                               client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None


if __name__ == "__main__":
    ingest()
