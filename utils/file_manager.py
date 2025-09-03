import os
import json
from PIL import Image


metadata_file_path = "metadata.json"


def get_metadata() -> list[dict]:
    if not os.path.exists(metadata_file_path):
        return []
    with open(metadata_file_path, "r") as f:
        return json.load(f)


def get_number_images() -> int:
    metadata = get_metadata()
    return len(metadata)


def save_metadata(metadata: list[dict]) -> None:
    # this works if file doesnt exist
    with open(metadata_file_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print("Image added to metadata")


def display_face_image(image_id: int) -> None:
    metadata = get_metadata()
    for image in metadata:
        if image["id"] == image_id:
            box = image["facial_area"]
            box = (box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"])
            img = Image.open(image["image_path"])
            img = img.crop(box)
            img.show()
            return
    print("Image not found")


if __name__ == "__main__":
    display_face_image(int(input("Enter image id: ")))
