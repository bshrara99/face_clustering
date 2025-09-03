from utils.file_manager import get_metadata, save_metadata
from utils.embeddor import create_embeddings
from utils.faiss_manager import FaissManager


def add_new_image(
    image_path: str,
    model_name: str,
    faiss_manager: FaissManager,
    metadata: list[dict],
    threshold: float
) -> None:
    # check if image already exists
    if image_path in [image["image_path"] for image in metadata]:
        print("Image already exists")
        return

    print("Creating embeddings...")
    representations = create_embeddings(image_path, model_name=model_name)
    print("Embeddings created")
    for i, representation in enumerate(representations):
        embedding = representation["embedding"]
        curr_new_image = {
            "id": len(metadata) + i + 1,
            "image_path": image_path,
            "facial_area": representation["facial_area"],
            "cluster_id": -1
        }
        # find nearest neighbors
        print("Searching for nearest neighbors...")
        distances, indices = faiss_manager.search(embedding, 1)
        distance, index = distances[0][0], indices[0][0]
        print("Nearest neighbor found:", distance, index)
        if distance > threshold or index == -1:
            max_cluster_id = max(
                [image["cluster_id"] for image in metadata]
                # if no images, return -1 so that first cluster id will be 0
                or [-1]
            )
            curr_new_image["cluster_id"] = max_cluster_id + 1
            print("Creating new cluster...")
        else:
            nearest_image = metadata[index]
            # add only if nearest image is different
            if nearest_image["image_path"] != curr_new_image["image_path"]:
                print("Adding to existing cluster ", nearest_image["cluster_id"])
                curr_new_image["cluster_id"] = nearest_image["cluster_id"]
        # add image to metadata
        print("Adding image to metadata...")
        metadata.append(curr_new_image)
        # add embedding to faiss index
        print("Adding embedding to faiss index...")
        faiss_manager.add_vectors(embedding.reshape(1, -1))

    # save metadata
    print("Saving metadata...")
    save_metadata(metadata)
