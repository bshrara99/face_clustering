import os
from utils.clusterer import add_new_image
from utils.file_manager import get_metadata
from utils.faiss_manager import FaissManager


metadata = get_metadata()
EMBEDDING_MODEL = "ArcFace"
EMBEDDING_SIZE = 512
print("Creating index...")
faiss_manager = FaissManager()
INDEX_PATH = "index.index"
if os.path.exists(INDEX_PATH):
    faiss_manager.load(INDEX_PATH)
else:
    faiss_manager.create_index(EMBEDDING_SIZE)
    faiss_manager.save(INDEX_PATH)
THRESHOLD = 15


image_path = input("Enter image path: ")
add_new_image(image_path, EMBEDDING_MODEL, faiss_manager, metadata, THRESHOLD)
faiss_manager.save(INDEX_PATH)
