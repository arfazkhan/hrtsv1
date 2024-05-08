import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import faiss

# Configuration
REFERENCE_IMAGE_PATH = '1.jpg'
EMBEDDING_FILE = 'reference_embeddings.npy'
THRESHOLD = 0.8

# Model Loading
face_detector = MTCNN(keep_all=True, device='cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# Preprocess Reference Image
ref_image = cv2.imread(REFERENCE_IMAGE_PATH)
if ref_image is None:
    print("Error: Could not read reference image.")
    exit()

ref_faces, _ = face_detector.detect(ref_image)

if ref_faces is not None and len(ref_faces) > 0:
    # Assuming only one face is detected
    x, y, w, h = [int(i) for i in ref_faces[0]]
    cropped_face = ref_image[y:y+h, x:x+w]
    input_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)  # Convert to RGB
    input_face = np.transpose(input_face, (2, 0, 1))  # Transpose dimensions
    input_face = torch.from_numpy(input_face).unsqueeze(0).float()  # Convert to tensor
    ref_embedding = facenet(input_face)
else:
    print("No face detected in reference image")
    exit()

# Save Embedding
np.save(EMBEDDING_FILE, ref_embedding.detach().numpy())

# Load or Precompute Reference Embeddings
try:
    ref_embeddings = np.load(EMBEDDING_FILE)
except FileNotFoundError:
    print("Error: Reference embeddings file not found.")
    exit()

# Build ANN Index
index = faiss.IndexFlatL2(ref_embeddings.shape[1])
index.add(ref_embeddings)

# Main Video Processing Loop
print("Starting video capture loop...")
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame from video capture.")
        break

    # Face Detection
    faces, _ = face_detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, w, h = [int(i) for i in face]
            cropped_face = frame[y:y+h, x:x+w]
            input_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)  # Convert to RGB
            input_face = np.transpose(input_face, (2, 0, 1))  # Transpose dimensions
            input_face = torch.from_numpy(input_face).unsqueeze(0).float()  # Convert to tensor

            # Fast Matching
            embedding = facenet(input_face)
            distances, indices = index.search(embedding.detach().numpy(), k=1)

            if distances[0][0] < THRESHOLD:
                name = 'Recognized'  # Replace with actual name association
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
