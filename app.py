%%writefile app.py
import streamlit as st
import cv2
import face_recognition
import numpy as np
from skimage import io
from skimage.transform import rescale
from matplotlib import pyplot as plt

# Initialize collection list to store faces
collection = []

def find_best_match(target_face, collection_faces):
    """
    Compare target face with collection faces and return the best match.
    """
    target_face_encoding = face_recognition.face_encodings(target_face)
    if len(target_face_encoding) == 0:
        return None, 0
    target_face_encoding = target_face_encoding[0]

    best_similarity = 0
    best_face = None
    for collection_face, collection_encoding in collection_faces:
        similarity = face_recognition.compare_faces([collection_encoding], target_face_encoding)
        if similarity[0] and similarity[0] > best_similarity:
            best_similarity = similarity[0]
            best_face = collection_face
    return best_face, best_similarity

def find_best_match_for_each_face(collection, target_faces, target_image):
    """
    Finds the best matching face for each collection face in the target image.
    """
    matches = []
    for collection_index, (collection_face, collection_encoding) in enumerate(collection):
        best_similarity = 0
        best_match_location = None

        for (x, y, w, h) in target_faces:
            target_face = target_image[y:y + h, x:x + w]
            best_face, similarity = find_best_match(target_face, [(collection_face, collection_encoding)])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_location = (x, y, w, h)

        if best_match_location:
            matches.append((best_match_location, best_similarity, collection_index))

    return matches

def process_image(uploaded_image):
    """
    Processes the uploaded image and adds detected faces to the collection.
    """
    global collection

    # Load the image using skimage.io
    image = io.imread(uploaded_image)

    # Resize if necessary
    if image.shape[0] > 4096 or image.shape[1] > 4096:
        image = rescale(image, 0.5, mode="constant")
        image = (image * 255).astype(np.uint8)
    
    # Detect faces using OpenCV's Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Add faces and embeddings to collection
    collection = []
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_embedding = face_recognition.face_encodings(face_rgb)
        if face_embedding:
            collection.append((face, face_embedding[0]))

    return image, faces

def main():
    st.title("Face Recognition Streamlit App")
    
    uploaded_image = st.file_uploader("Upload an image with faces", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        image, faces = process_image(uploaded_image)

        # Show the uploaded image with detected faces
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Show detected faces individually
        st.subheader("Detected Faces")
        for i, (face, _) in enumerate(collection):
            st.image(face, caption=f"Face {i+1}", use_column_width=True)

        # Upload the target image
        target_image = st.file_uploader("Upload a target image for face matching", type=["jpg", "png", "jpeg"])
        if target_image is not None:
            target_image = io.imread(target_image)
            target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
            target_faces = face_cascade.detectMultiScale(target_gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            # Find best matches for each face in the collection
            matches = find_best_match_for_each_face(collection, target_faces, target_image)

            # Draw bounding boxes for matches
            for (x, y, w, h), _, collection_index in matches:
                cv2.rectangle(target_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Display the target image with matched faces
            st.image(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB), caption="Target Image with Matched Faces", use_column_width=True)
            
            # Optionally, display detailed matching results
            st.subheader("Matching Results")
            for (x, y, w, h), similarity, collection_index in matches:
                st.write(f"Face {collection_index + 1}: Match found at (x={x}, y={y}) with similarity {similarity:.2f}")

if __name__ == "__main__":
    main()
