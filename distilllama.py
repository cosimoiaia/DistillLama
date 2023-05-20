"""
Mainly based on:
https://python.langchain.com/en/latest/modules/agents/agent_executors/examples/agent_vectorstore.html

"""
import os
import glob
import shutil
from typing import List, Any, Dict
from dotenv import load_dotenv
from prompt_toolkit import prompt

from langchain.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

from pydantic import BaseModel
from langchain.embeddings.base import Embeddings

import uuid
import os

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')

persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
HF_EMB = os.environ.get('HF_EMB')

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {".txt": (TextLoader, {"encoding": "utf8"})}

load_dotenv()

try:
    from llama_cpp import Llama
except ImportError as exc:
    raise ValueError(
        "Could not import llama_cpp python package. Please install it with `pip install llama_cpp`.") from exc

if HF_EMB:
    # HuggingFace embeddings have a lower acc but much higher speed
    from langchain.embeddings import HuggingFaceEmbeddings

    embed = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    model = None
else:
    model = Llama(model_path=model_path, embedding=True, verbose=False)


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

        self.model = model  # Llama(model_path=self.model_path, embedding=True, verbose=False)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # print(f"Embedding documents: {texts}")
        embeddings = [self.model.create_embedding(t)['data'][0]['embedding'] for t in texts]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        # print("Embedding query")
        embeddings = self.model.create_embedding(text)['data'][0]['embedding']
        return embeddings


def load_from_text(text: str) -> Document:
    return Document(page_content=text, metadata={})


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

    processed_directory = os.path.join(source_dir, 'processed')
    os.makedirs(processed_directory, exist_ok=True)

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
        source_file_path = file_path
        destination_file_path = os.path.join(processed_directory, os.path.basename(file_path))
        shutil.move(source_file_path, destination_file_path)

    progress_bar.close()

    return results


chunk_size = 512
chunk_overlap = 50


def ingest(source_dir: str = source_directory):
    print(f"Ingesting documents from {source_dir}")
    documents = load_documents(source_dir)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {source_directory}")
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} characters each)")

    db = Chroma.from_documents(texts, embed, persist_directory=persist_directory,
                               client_settings=CHROMA_SETTINGS)
    db.persist()

    db = None


def query():
    db = Chroma(persist_directory=persist_directory, embedding_function=embed,
                client_settings=CHROMA_SETTINGS)

    retriever = db.as_retriever()

    callbacks = [StreamingStdOutCallbackHandler()]
    llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False, n_gpu_layers=15)
    runner = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    while True:
        q = prompt("\nEnter your question, type 'Information' to provide additional "
                   "information to the model or type 'quit' to exit\n> ")
        if q == "quit":
            break
        if q == "Information":
            text = prompt("\nEnter the information you want to distill into your A.I. \n> ")
            print("Ingesting new informations...", end='')
            db.add_texts(texts=[text], ids=str(uuid.uuid1()))
            print("Done")
            continue

        res = runner(q)
        answer, docs = res['result'], res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(q)
        print("\n> Answer:")
        print(answer)

        # Print the sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)


if __name__ == "__main__":
    ingest()
    query()
