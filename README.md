# FaceMatcher


This project demonstrates how to use OpenCV and the `face_recognition` library to detect and recognize faces in images. The system allows users to create a face collection, add detected faces, match faces with a target image, and visualize the results by drawing bounding boxes around the faces.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)

## Project Overview

This project provides a simple pipeline for working with face detection and recognition:

1. **Face Collection**: Create a collection to store detected faces and their embeddings.
2. **Face Detection**: Detect faces in uploaded images using Haar Cascade Classifiers from OpenCV.
3. **Face Matching**: Match faces in a target image to those in the collection using similarity scores.
4. **Bounding Boxes**: Visualize face detections and matches with bounding boxes.
5. **Collection Management**: Add and remove faces from the collection as needed.

The project uses popular Python libraries such as OpenCV for face detection and `face_recognition` for recognizing and comparing faces.

## Installation

To get started with the project, follow these simple steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/MahmoudSaad21/FaceMatcher.git
   cd FaceMatcher
   ```

2. **Set up a virtual environment (optional but recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Upload an Image**: Upload an image to be processed by the system. If the image is too large, it will be resized to fit within the system's processing limits.

2. **Add Faces to the Collection**: The system will detect faces in the uploaded image and add them to the collection with their embeddings.

3. **Match Faces**: After adding faces to the collection, you can upload a target image to compare and match the faces in the collection with those in the target image.

4. **Visualize Results**: The system will display bounding boxes around the detected and matched faces in both the original and target images.

5. **Clear the Collection**: Once you're done, you can clear the face collection to free up memory.

## Project Structure

```
face-recognition-collection/
│
├── requirements.txt          # Python dependencies
├── face_recognition.ipynp    # Main Notebook for face detection and recognition
├── images/                   # Directory for input images
├── README.md                 # Project documentation
```

## Dependencies

This project uses the following libraries:

- `opencv-python` (for face detection)
- `face_recognition` (for recognizing and comparing faces)
- `skimage` (for image loading and scaling)
- `matplotlib` (for image visualization)
- `numpy` (for numerical operations)

To install all dependencies, run:

```bash
pip install -r requirements.txt
```
