from . import utils
import os
import numpy as np
import json
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rerankers import Reranker
from langchain.chat_models import init_chat_model


class Embedder:
    """Handles file loading, chunking and text embedding and storing embeddings in FAISS & JSON."""
    
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.data_path = os.getenv("data_path", "")
        self.vector_embedding_path = os.getenv("embedding_path", "") + 'chunk_vectors.faiss'
        self.chunkdata_path = os.getenv("embedding_path", "") + 'chunk_data.json'
        self.index = None
        self.chunkdata = None
        self._load_embeddings()
        
    def _load_embeddings(self):
        """Loads embeddings and chunk data from files if they exist."""
        chunk_file_exists = os.path.exists(self.chunkdata_path)
        vector_file_exists = os.path.exists(self.vector_embedding_path)
        if chunk_file_exists and vector_file_exists:
            self._load_index()
            self._load_chunkdata()
        elif chunk_file_exists and not vector_file_exists:
            self._load_index_from_chunkdata(from_disk=True, to_disk=True)
        else:
            return

    def _load_index_from_chunkdata(self, from_disk=True, to_disk=True):
        """Creates a new FAISS index from chunk data if the vector embedding file is missing."""
        if from_disk:
            self._load_chunkdata()      
        self._init_index()

        chunk_vectors = self.encode([chunk["text"] for chunk in self.chunkdata])
        self.index.add(chunk_vectors)

        if to_disk:
            self._delete_embedding_files(keep="chunk")
            self._save_index_file()

    def _delete_embedding_files(self, keep=None):
        """Deletes the vector embedding and chunk data files."""
        if os.path.exists(self.vector_embedding_path) and keep != "vector":
            os.remove(self.vector_embedding_path)
        if os.path.exists(self.chunkdata_path) and keep != "chunk":
            os.remove(self.chunkdata_path)
    
    def _load_documents(self):
        """Loads text data from the data_path."""
        docs = []
        for filename in os.listdir(self.data_path):
            file_path = os.path.join(self.data_path, filename)
            if filename.endswith(".pdf"):
                docs.append({"filename": filename, "text": pdf_to_text(file_path)})
            elif filename.endswith(".txt"):
                docs.append({"filename": filename, "text": txt_to_text(file_path)})
            elif os.path.isdir(file_path):
                continue
            else:
                print(f"Could not load file {filename}")
        return docs

    def _chunk_documents(self, chunk_size: int, chunk_overlap: int):
        """Splits texts into chunks."""
        docs = self._load_documents()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = []
        for doc in docs:
            chunk_texts = splitter.split_text(doc["text"])
            filename = doc["filename"]
            for i, chunk_text in enumerate(chunk_texts):
                chunks.append({"filename": filename, "text": chunk_text, "vector_id": i})

        self.chunkdata = chunks

    def encode(self, texts: list[str]) -> np.ndarray:
        """Returns vector embeddings for a list of texts."""
        return self.model.encode(texts, convert_to_numpy=True)
    
    def _init_index(self):
        """Creates FAISS index and stores it in the vector_embedding_path."""
        self.index = faiss.IndexFlatL2(self.dim)

    def _load_index(self):
        """Loads FAISS index from the vector_embedding_path."""
        self.index = faiss.read_index(self.vector_embedding_path)
    
    def _save_index_file(self):
        """Saves FAISS index to the vector_embedding_path."""
        faiss.write_index(self.index, self.vector_embedding_path)
    
    def _init_chunkdata(self):
        """Creates chunk data file if it does not exist."""
        self.chunkdata = None
        if not os.path.exists(self.chunkdata_path):
            with open(self.chunkdata_path, "w") as f:
                json.dump([], f)

    def _load_chunkdata(self):
        """Loads chunk data from JSON file if it exists; otherwise, returns an empty list."""
        try:
            with open(self.chunkdata_path, "r") as f:
                self.chunkdata = json.load(f)
        except json.JSONDecodeError:
            pass

    def _save_chunkdata_file(self):
        """Saves chunk data to JSON file."""
        with open(self.chunkdata_path, "w") as f:
            json.dump(self.chunkdata, f, indent=4)

    def search(self, query_text, top_k=3):
        """Searches FAISS and retrieves text chunk data from JSON."""
        query_vector = self.encode([query_text])
        distances, indices = self.index.search(query_vector, top_k)
        return distances, indices
    
class RAG:
    def __init__(self,
                 embedding_model_name="all-MiniLM-L6-v2", LLM_name="openai:gpt-4o-mini",
                 reranker_name="flashrank", use_reranker=True,
                 retrieve_top_k=15, rerank_top_k=3):       
        self.data_path = os.getenv("data_path")
        self.vector_embedding_path = os.getenv("vector_embedding_path")
        
        self.embedder = Embedder(embedding_model_name)
        self.use_reranker = use_reranker
        self.reranker = Reranker(reranker_name) if use_reranker else None
        self.LLM = init_chat_model(LLM_name)

        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k

    def reload_embeddings(self, chunk_size=500, chunk_overlap=50):
        # check if data path contains files
        if len(os.listdir(self.data_path)) == 0:
            print("No files found in data_path.")
            return
        
        self.embedder._delete_embedding_files()
        self.embedder._init_index()
        self.embedder._init_chunkdata()
        self.embedder._chunk_documents(chunk_size, chunk_overlap)
        self.embedder._load_index_from_chunkdata(from_disk=False, to_disk=True)
        self.embedder._save_chunkdata_file()

    def _retrieve_context(self, query, top_k):
        distances, indices = self.embedder.search(query, top_k=top_k)
        relevant_chunks = [self.embedder.chunkdata[idx] for idx in indices[0]]
        for i, chunk in enumerate(relevant_chunks):
            chunk["distance"] = distances[0][i]
        return relevant_chunks
    
    def _rerank_chunks(self, query, chunks, top_k):
        # convert chunk format to list of texts
        chunk_texts = [chunk["text"] for chunk in chunks]
        chunk_ids = [chunk["vector_id"] for chunk in chunks]

        reranked_chunks = self.reranker.rank(query=query, docs=chunk_texts, doc_ids=chunk_ids)
        top_k_results = reranked_chunks.results[:top_k]
        output_chunks = []
        for r in top_k_results:
            for chunk in chunks:
                if chunk['vector_id'] == r.doc_id:
                    chunk['score'] = r.score
                    output_chunks.append(chunk)

        return output_chunks

    def retrieve(self, query):
        retrieved_chunks = self._retrieve_context(query, top_k=self.retrieve_top_k)

        if self.use_reranker:
            chunks = self._rerank_chunks(query, retrieved_chunks, top_k=self.rerank_top_k)
        else:
            chunks = retrieved_chunks[:self.rerank_top_k]

        return chunks

    def query(self, query):
        chunks = self.retrieve(query)
        
        joined_chunks = " ".join([chunk["text"] for chunk in chunks])
        
        # Format the prompt
        prompt = f"""
        You are an AI assistant. Use the following retrieved context to answer the question.

        Context:
        {joined_chunks}

        Question:
        {query}
        """

        answer = self.query_LLM(prompt)

        response = {
            "chunks": chunks,
            "query": query,
            "answer": answer
        }	

        return response
    
    def query_LLM(self, query):
        response = self.LLM.invoke(query).content
        return response
    
    def print_chunk(self, id):
        print(self.embedder.chunkdata[id]['text'])