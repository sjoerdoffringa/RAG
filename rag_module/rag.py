from .utils import pdf_to_text, txt_to_text
from .prompts import prompts

import os
from typing import List
import numpy as np
import json
import faiss
#import pickle
import ast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rerankers import Reranker
from langchain.chat_models import init_chat_model
import mlflow
from sklearn.metrics.pairwise import cosine_similarity

from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi


class Embedder:
    """Handles file loading, chunking and text embedding and storing embeddings in FAISS & JSON."""
    
    def __init__(self, model_name: str, normalize_embeddings: bool = True):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.normalize_embeddings = normalize_embeddings
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
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=self.normalize_embeddings)
    
    def _init_index(self) -> None:
        """Creates FAISS index and stores it in the vector_embedding_path."""
        self.index = faiss.IndexFlatL2(self.dim)

    def _load_index(self) -> None:
        """Loads FAISS index from the vector_embedding_path."""
        self.index = faiss.read_index(self.vector_embedding_path)
    
    def _save_index_file(self) -> None:
        """Saves FAISS index to the vector_embedding_path."""
        faiss.write_index(self.index, self.vector_embedding_path)
    
    def _init_chunkdata(self) -> None:
        """Creates chunk data file if it does not exist."""
        self.chunkdata = None
        if not os.path.exists(self.chunkdata_path):
            with open(self.chunkdata_path, "w") as f:
                json.dump([], f)

    def _load_chunkdata(self) -> None:
        """Loads chunk data from JSON file if it exists; otherwise, returns an empty list."""
        try:
            with open(self.chunkdata_path, "r", encoding="utf-8") as f:
                self.chunkdata = json.load(f)
        except json.JSONDecodeError:
            pass

    def _save_chunkdata_file(self) -> None:
        """Saves chunk data to JSON file."""
        with open(self.chunkdata_path, "w") as f:
            json.dump(self.chunkdata, f, indent=4)

    def search(self, query_text, top_k=3) -> tuple[np.ndarray, np.ndarray]:
        """Searches FAISS and retrieves text chunk data from JSON."""
        query_vector = self.encode([query_text])
        distances, indices = self.index.search(query_vector, top_k)
        return distances, indices

class hybridEmbedder(Embedder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bm25 = None
        self.corpus = None

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes and lowercases text consistently for both corpus and queries."""
        return word_tokenize(text.lower())

    def _build_corpus(self):
        """Tokenizes the corpus for BM25."""
        self.corpus = [self._tokenize(chunk['text']) for chunk in self.chunkdata]

    def _build_bm25(self):
        """Initializes BM25 from tokenized corpus."""
        self._build_corpus()
        self.bm25 = BM25Okapi(self.corpus)

    def search_bm25(self, query: str, top_k: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs a BM25-based search and returns:
        - distances: array of BM25 scores (float)
        - indices: array of corresponding indices into the corpus
        """
        if self.bm25 is None:
            self._build_bm25()

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]

        return np.array(top_scores), np.array(top_indices)

class RAG:
    def __init__(self,
                 embedding_model_name="all-MiniLM-L6-v2", normalize_embeddings=True,
                 LLM_name="openai:gpt-4o-mini",
                 reranker_name="flashrank", use_reranker=True,
                 retrieve_top_k=15, rerank_top_k=3,
                 scope_model_id=None):
        
        self.embedder = hybridEmbedder(embedding_model_name, normalize_embeddings=normalize_embeddings)
        self.use_reranker = use_reranker
        self.reranker = Reranker(reranker_name) if use_reranker else None
        self.LLM = init_chat_model(LLM_name)

        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k

        self.scope_model_id = scope_model_id
        self.scope_model = None
        self.scope_features = []
        if scope_model_id is not None:
            self._load_scope_model()

    def _load_scope_model(self):
        # Construct the model URI
        model_uri = f"runs:/{self.scope_model_id}/model"

        # Load the model
        try:
            model = mlflow.sklearn.load_model(model_uri)
            print(f"Successfully loaded model from run {self.scope_model_id}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.scope_model = model

        # Retrieve features from the run params
        try:
            run = mlflow.get_run(self.scope_model_id)
            features_str = run.data.params.get("features", "[]")
            self.scope_features = ast.literal_eval(features_str)
            print(f"Loaded features: {self.scope_features}")
        except Exception as e:
            print(f"Error loading features from run: {e}")
            self.scope_features = []

    def reload_embeddings(self, chunk_size=500, chunk_overlap=50):
        # check if data path contains files
        if len(os.listdir(self.embedder.data_path)) == 0:
            print("No files found in data_path.")
            return
        
        self.embedder._delete_embedding_files()
        self.embedder._init_index()
        self.embedder._init_chunkdata()
        self.embedder._chunk_documents(chunk_size, chunk_overlap)
        self.embedder._load_index_from_chunkdata(from_disk=False, to_disk=True)
        self.embedder._save_chunkdata_file()

    def _retrieve_dense(self, query, top_k):
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
        retrieved_chunks = self._retrieve_dense(query, top_k=self.retrieve_top_k)

        if self.use_reranker:
            chunks = self._rerank_chunks(query, retrieved_chunks, top_k=self.rerank_top_k)
        else:
            chunks = retrieved_chunks[:self.rerank_top_k]

        return chunks
    
    def _format_prompt(self, query, chunks, prompt="vanilla-rag", add_filename=True):
        if add_filename:
            chunklist = [f'{chunk["filename"]}: {chunk["text"]}' for chunk in chunks]
        else:
            chunklist = [chunk["text"] for chunk in chunks]

        if prompt == "self-citation":
            chunklist = [f'[{i}] {doc}' for i, doc in enumerate(chunklist)]

        joined_chunks = " ".join(chunklist)
        formatted_prompt = prompts[prompt].format(joined_chunks=joined_chunks, query=query)
        return formatted_prompt
    
    def predict_scope(self, chunks, query):
        best_chunk = chunks[0]

        if 'top_sparse_score' in self.scope_features:
            top_sparse_score = self.embedder.search_bm25(query, top_k=1)[0][0]

        features = []
        for f in self.scope_features:
            if f == 'top_sparse_score':
                features.append(top_sparse_score)
            elif f == 'top_reranker_score':
                features.append(best_chunk['score'])
            elif f == 'top_dense_score':
                features.append(best_chunk['distance'])
            else:
                raise ValueError(f"Unknown feature: {f}")
        print(f"Features for scope prediction: {features}")
        prediction = self.scope_model.predict(np.array(features).reshape(1, -1))[0]
        return prediction
    
    def query_LLM(self, query):
        response = self.LLM.invoke(query).content
        return response

    def query(self, query):
        # get relevant chunks
        chunks = self.retrieve(query)

        # format the response
        response = {
            "chunks": chunks,
            "query": query,
        }

        prompt_format = "vanilla-rag"

        # predict if query falls within the scope of the database
        if self.scope_model is not None:
            scope_pred = self.predict_scope(chunks, query)
            response["scope_prediction"] = scope_pred

            if scope_pred == 0:
                prompt_format = "self-citation"
            else:
                counterfactual_prompt = self._format_prompt(query, chunks, prompt="counterfactual")
                counterfactual_response = self.query_LLM(counterfactual_prompt)
                response["counterfactual"] = counterfactual_response          

        # query the LLM
        prompt = self._format_prompt(query, chunks, prompt=prompt_format)
        answer = self.query_LLM(prompt)
        response["answer"] = answer

        return response

    
    def print_chunk(self, id):
        print(self.embedder.chunkdata[id]['text'])
        