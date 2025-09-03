import os
from PIL import Image 
import streamlit as st
from utils.clusterer import add_new_image
from utils.file_manager import get_metadata
from utils.faiss_manager import FaissManager


# initialize
EMBEDDING_MODEL = "ArcFace"
EMBEDDING_SIZE = 512
THRESHOLD = 15
INDEX_PATH = "index.index"
if "metadata" not in st.session_state:
    st.session_state.metadata = get_metadata()
if "faiss_manager" not in st.session_state:
    print("Creating index...")
    st.session_state.faiss_manager = FaissManager()
    if os.path.exists(INDEX_PATH):
        st.session_state.faiss_manager.load(INDEX_PATH)
    else:
        st.session_state.faiss_manager.create_index(EMBEDDING_SIZE)
        st.session_state.faiss_manager.save(INDEX_PATH)
    print("Index created")



# config
st.set_page_config(
    page_title="Face Clustering App",
    page_icon=":camera:",
    layout="wide",
    initial_sidebar_state="expanded"
)


# header
st.markdown(
    "<h1 style='text-align: center; color: #2196F3; margin-bottom: 20px;'>"
    "👦🏻 Face Clustering App"
    "</h1>",
    unsafe_allow_html=True
)


# file uploader
file = st.file_uploader(
    "choose an image...",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=False
)


st.divider()


st.title("🖼️ Folders:")
if 'storage_dir' not in st.session_state:
    st.session_state.storage_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'storage'
    )
    if not os.path.exists(st.session_state.storage_dir):
        os.makedirs(st.session_state.storage_dir)


if 'selected_cluster' not in st.session_state:
    max_cluster_id = max([image["cluster_id"] for image in st.session_state.metadata] or [-1])
    st.session_state.clusters = [i for i in range(max_cluster_id + 1)]
    st.session_state.selected_cluster = None


# display folders
if 'clusters' in st.session_state and len(st.session_state.clusters) > 0:
    num_cols = 3
    num_rows = -(-len(st.session_state.clusters) // num_cols)
    cols = st.columns(num_cols)
    for i in range(num_rows):
        for j in range(num_cols):
            z = (i * num_cols) + j
            if z >= len(st.session_state.clusters):
                break
            cluster_id = st.session_state.clusters[z]
            with cols[j]:
                if st.button(f"📁Cluster {cluster_id}", key=cluster_id):
                    st.session_state.selected_cluster = cluster_id
else:
    st.write("No folders found")


# display folder images
if st.session_state.selected_cluster is not None:
    images_paths = [
        image_info["image_path"] for image_info in st.session_state.metadata
        if image_info["cluster_id"] == st.session_state.selected_cluster
    ]
    for i, image_path in enumerate(images_paths):
        img = Image.open(image_path)
        with cols[i % 3]:
            st.image(img, caption=None, use_container_width=True)


st.divider()


# handle image input
def image_clustering(image_path: str) -> bool:
    try:
        if image_path is not None:
            add_new_image(
                image_path=image_path,
                model_name=EMBEDDING_MODEL,
                faiss_manager=st.session_state.faiss_manager,
                metadata=st.session_state.metadata,
                threshold=THRESHOLD
            )
            st.success("Image added successfully")
            st.session_state.faiss_manager.save(INDEX_PATH)
            return True
        else:
            return False
    except Exception as e:
        st.error(e)
        return False
def save_image(file):
    # save image to storage
    new_path = os.path.join(st.session_state.storage_dir, file.name)
    with open(new_path, "wb") as f:
        f.write(file.getbuffer())
    return new_path

if file is not None:
    new_path = save_image(file)
    # add image to corresponding cluster
    image_clustering(new_path)
