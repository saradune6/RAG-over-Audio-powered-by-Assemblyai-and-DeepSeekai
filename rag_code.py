from qdrant_client import models
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import assemblyai as aai
from typing import List, Dict
import groq 
import time



from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)

def batch_iterate(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

class EmbedData:

    def __init__(self, embed_model_name="BAAI/bge-large-en-v1.5", batch_size=32):
        self.embed_model_name = embed_model_name
        self.embed_model = self._load_embed_model()
        self.batch_size = batch_size
        self.embeddings = []

    def _load_embed_model(self):
        return HuggingFaceEmbedding(model_name=self.embed_model_name, trust_remote_code=True, cache_folder='./hf_cache')

    def generate_embedding(self, context):
        return self.embed_model.get_text_embedding_batch(context)

    def embed(self, contexts):
        self.contexts = contexts
        for batch_context in batch_iterate(contexts, self.batch_size):
            batch_embeddings = self.generate_embedding(batch_context)
            self.embeddings.extend(batch_embeddings)

class QdrantVDB_QB:

    def __init__(self, collection_name, vector_dim=768, batch_size=512):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_dim = vector_dim

    def define_client(self):
        self.client = QdrantClient(url="http://localhost:6333", prefer_grpc=True)

    def create_collection(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.vector_dim, distance=models.Distance.DOT, on_disk=True),
                optimizers_config=models.OptimizersConfigDiff(default_segment_number=5, indexing_threshold=0),
                quantization_config=models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True)),
            )

    def ingest_data(self, embeddata):
        for batch_context, batch_embeddings in zip(batch_iterate(embeddata.contexts, self.batch_size), 
                                                   batch_iterate(embeddata.embeddings, self.batch_size)):
            self.client.upload_collection(collection_name=self.collection_name,
                                          vectors=batch_embeddings,
                                          payload=[{"context": context} for context in batch_context])

        self.client.update_collection(collection_name=self.collection_name,
                                      optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000))

class Retriever:

    def __init__(self, vector_db, embeddata):
        self.vector_db = vector_db
        self.embeddata = embeddata

    def search(self, query):
        query_embedding = self.embeddata.embed_model.get_query_embedding(query)
        result = self.vector_db.client.search(
            collection_name=self.vector_db.collection_name,
            query_vector=query_embedding,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(ignore=False, rescore=True, oversampling=2.0),
            ),
            timeout=1000,
        )
        return result

class RAG:
    def __init__(self, retriever, model_name="deepseek-r1-distill-qwen-32b"):
        self.groq_client = groq.Client(api_key="api_key")
        self.model_name = model_name   # Replace with your actual API key
        self.retriever = retriever
        self.qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context}\n"
            "---------------------\n"
            "Given the context information above, think step by step to answer the query concisely."
            " If you don't know the answer, say 'I don't know!'.\n"
            "Query: {query}\n"
            "Answer: "
        )

    def generate_context(self, query):
        result = self.retriever.search(query)
        if not result:
            print("No relevant context found.")  # Debugging output
            return "No relevant context found."
        
        context = [dict(data) for data in result]
        combined_prompt = [entry["payload"]["context"] for entry in context[:2]]
        
        print("Retrieved Context:\n", combined_prompt)  # Debugging output
        return "\n\n---\n\n".join(combined_prompt)

        
    def query(self, query, max_retries=5):
        context = self.generate_context(query=query)
        prompt = self.qa_prompt_tmpl_str.format(context=context, query=query)

        print("Generated Prompt:\n", prompt)  # Debugging output

        delay = 2
        for attempt in range(max_retries):
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                print("Raw Response:\n", response)  # Debugging output
                return response.choices[0].message.content
            except groq.InternalServerError as e:
                print(f"Error encountered: {e}, retrying... (Attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2  

    

class Transcribe:
    def __init__(self, api_key: str = None):
        """Initialize the Transcribe class with AssemblyAI API key."""
        aai.settings.api_key = 'api_key'
        self.transcriber = aai.Transcriber()

    def transcribe_audio(self, audio_path: str) -> List[Dict[str, str]]:
        """
        Transcribe an audio file and return speaker-labeled transcripts.
        """
        config = aai.TranscriptionConfig(speaker_labels=True, speakers_expected=2)
        transcript = self.transcriber.transcribe(audio_path, config=config)
        return [{"speaker": f"Speaker {utt.speaker}", "text": utt.text} for utt in transcript.utterances]
