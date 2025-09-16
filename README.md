# Faces Clusterer
This project aims to provide a simple interface to save images and cluster them based on faces.


## Project Structure:
- Embeddor: recognizes faces within images and creates embeddings vectors for them
- Faiss Manager: provides a class o interact with FAISS vector db easily
- Clusterer: provides the core method `add_new_image` which:
    - extracts & embdes faces (using embeddor)
    - search db for nearest vector for each face
    - decides whether to add each face to an existing cluster or create new one
    - save new vectors to db & images info to metadata
- File Manager: provides utility functions to interact with metadata


## Interface:
[App.py](./app.py) provides a streamlit interface that allows the user to upload images.
Once the user uplaods an image, it adds it to the storage and clusters it.
The user can see the differnt clusters folders, and see images inside each folder.


## Usage:
Assuming you have git and python installed, kindly follow the steps below:
- Clone the repository:
```
git clone https://github.com/bshrara99/face_clustering.git
```

- Install dependencies:
```
pip install -r requirements.txt
```

- Run app:
```
streamlit run app.py
```
