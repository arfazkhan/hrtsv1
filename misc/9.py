import cv2
import numpy as np
import faiss
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# *** Configuration - Adjust as needed ***
REFERENCE_IMAGE_PATH = '1.jpg'
EMBEDDING_FILE = 'reference_embeddings.pkl'
THRESHOLD = 0.8 
DETECTION_DOWNSAMPLE = 2  # Downsample for faster face detection

# *** Load Quantized Models (Assuming you've quantized them)***
face_detector = MTCNN(keep_all=True, device='cpu', select_largest=False) 
face_detector.load_state_dict(torch.load('best_pnet.pth'))
facenet = InceptionResnetV1(pretrained='vggface2').eval()
facenet.load_state_dict(torch.load('facenet_quantized.pth'))

# *** Preprocess Reference Image***
ref_image = cv2.imread(REFERENCE_IMAGE_PATH)
ref_faces, _ = face_detector(cv2.resize(ref_image, (0, 0), fx=1/DETECTION_DOWNSAMPLE, fy=1/DETECTION_DOWNSAMPLE))

if ref_faces is not None:
    ref_embedding = facenet(ref_faces[0].unsqueeze(0))  # Assume single face
else:
    print("No face detected in reference image")
    exit()

# *** Optional: Compress and Store Embedding ***
# ... (Example using simple numpy save)
np.save(EMBEDDING_FILE, ref_embedding.detach().numpy())

# *** Load or Precompute Reference Embeddings ***
try:
    ref_embeddings = np.load(EMBEDDING_FILE)
except FileNotFoundError:
    # Compute embedding if not pre-stored (do this once then save)
    pass

# *** Build ANN Index ***
index = faiss.IndexFlatL2(ref_embeddings.shape[1])  
index.add(ref_embeddings)

# *** Main Video Processing Loop ***
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Optimized Face Detection
    small_frame = cv2.resize(frame, (0, 0), fx=1/DETECTION_DOWNSAMPLE, fy=1/DETECTION_DOWNSAMPLE)
    faces, _ = face_detector(small_frame)

    if faces is not None:
        for face in faces:
            x, y, w, h = face['box']
            x *= DETECTION_DOWNSAMPLE
            y *= DETECTION_DOWNSAMPLE
            w *= DETECTION_DOWNSAMPLE
            h *= DETECTION_DOWNSAMPLE

            cropped_face = frame[y:y+h, x:x+w]

            # FaceNet Embedding (on the original resolution)
            embedding = facenet(cropped_face.unsqueeze(0))

            # Fast Matching
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
