<<<<<<< HEAD
import os

# Hugging Face API Key
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Paths
PERSIST_DIR = "chroma_db"
DOCS_DIR = "data"

# Model & embedding settings
# You can use any Hugging Face models you like
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Or "HuggingFaceH4/zephyr-7b-beta"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval
RETRIEVAL_K = 3
=======
import os

# Hugging Face API Key
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Paths
PERSIST_DIR = "chroma_db"
DOCS_DIR = "data"

# Model & embedding settings
# You can use any Hugging Face models you like
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Or "HuggingFaceH4/zephyr-7b-beta"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval
RETRIEVAL_K = 3
>>>>>>> 714e7ee4806695fa14480467dd7ec16818e9ee2c
