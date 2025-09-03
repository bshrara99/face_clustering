import numpy as np
from PIL import Image
from deepface import DeepFace


def create_embeddings(image_path: str, model_name: str) -> list[dict]:
    results = DeepFace.represent(
        img_path=image_path,
        model_name=model_name,
        enforce_detection=False
    )
    facial_areas = []
    # convert embedding lists to np array
    for i, result in enumerate(results):
        results[i]["embedding"] = np.array(result["embedding"]).reshape(1, -1)
        face_area = result["facial_area"]["w"] * result["facial_area"]["h"]
        facial_areas.append(face_area)
    # this is a hacky way to get rid of some false positives of the model
    # sometimes it will detect a face that is not there, usually it's too small
    # compare face areas to exclude outliers
    # we are using max not mean coz mean is affected by outliers
    # those models usually dont do false positives of large sizes so its ok to use max
    max_face_area = np.max(facial_areas)
    # this ratio is arbitrary chosen after experimenting
    # this isnt a good solution tho but better than nothing
    # it trades off some false negative to avoid false positives
    face_relative_size_threshold = 3
    results = [
        result
        for i, result in enumerate(results)
        # exclude faces that are too small compared to the largest face
        if facial_areas[i] > max_face_area / face_relative_size_threshold
    ]
    return results


def get_image_area(image_path: str) -> float:
    img = Image.open(image_path)
    return img.size[0] * img.size[1]
