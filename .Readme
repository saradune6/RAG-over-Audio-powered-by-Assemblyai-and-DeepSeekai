# RAG-over-Audio-powered-by-Assemblyai-and-DeepSeekai  


This project builds a RAG app over audio files.
We use:
- AssemblyAI to generate transcripts from audio files.
- LlamaIndex for orchestrating the RAG app.
- Qdrant VectorDB for storing the embeddings.
- Streamlit to build the UI.
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

## Installation and setup

**Setup AssemblyAI**:

Get an API key from [AssemblyAI](http://bit.ly/4bGBdux) and set it in the `.env` file as follows:

```bash
ASSEMBLYAI_API_KEY=<YOUR_API_KEY> a
```

**Setup SambaNova**:

Get an API key from [SambaNova](https://sambanova.ai/) and set it in the `.env` file as follows:

```bash
SAMBANOVA_API_KEY=<YOUR_SAMBANOVA_API_KEY> 
```

Note: Instead of SambaNova, you can also use Ollama.

**Setup Qdrant VectorDB**
   ```bash
   docker run -p 6333:6333 -p 6334:6334 \
   -v $(pwd)/qdrant_storage:/qdrant/storage:z \
   qdrant/qdrant
   ```

**Install Dependencies**:
   Ensure you have Python 3.11 or later installed.
   ```bash
   pip install streamlit assemblyai llama-index-vector-stores-qdrant llama-index-llms-sambanovasystems sseclient-py
   ```

**Run the app**:

   Run the app by running the following command:

   ```bash
   streamlit run app.py
   ```

---
