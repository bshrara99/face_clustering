import faiss
import numpy as np


class FaissManager:
    def __init__(self) -> None:
        self.index = None

    def create_index(self, dimensions: int) -> None:
        self.index = faiss.IndexFlatL2(dimensions)

    def add_vectors(self, vectors: np.ndarray) -> None:
        if self.index is None:
            self.create_index(vectors.shape[1])
        self.index.add(vectors)

    def search(
        self, vectors: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise Exception("Index not created")
        if vectors.shape[1] != self.index.d:
            raise Exception("Dimension mismatch")
        return self.index.search(vectors, k)

    def save(self, index_path: str) -> None:
        if self.index is None:
            raise Exception("Index not created")
        faiss.write_index(self.index, index_path)

    def load(self, index_path: str) -> None:
        if self.index is not None:
            raise Exception("Index already created")
        self.index = faiss.read_index(index_path)
