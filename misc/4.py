import cv2
import numpy as np
import os
from keras_facenet import FaceNet

# Load FaceNet model
facenet = FaceNet()

# Load reference images from folder
reference_images = []
reference_embeddings = []
reference_folder = "face"
for filename in os.listdir(reference_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = cv2.imread(os.path.join(reference_folder, filename))
        embedding = facenet.extract(img, threshold=0.95)
        reference_images.append(img)
        reference_embeddings.append(embedding)

# Load SSD face detection model
face_detection_model = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize frame to 300x300 for face detection
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )

    # Detect faces in the frame
    face_detection_model.setInput(blob)
    detections = face_detection_model.forward()

    # Loop through detected faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Get coordinates of bounding box
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract face ROI
            face = frame[startY:endY, startX:endX]

            # Perform face recognition using FaceNet
            embedding = facenet.extract(face, threshold=0.95)

            # Compare embeddings with reference embeddings
            best_match_idx = -1
            min_distance = float('inf')
            for idx, ref_embedding in enumerate(reference_embeddings):
                ref_embedding_np = np.array(ref_embedding)  # Convert to NumPy array
                embedding_np = np.array(embedding)  # Convert to NumPy array
                distance = np.linalg.norm(embedding_np - ref_embedding_np)
                if distance < min_distance:
                    min_distance = distance
                    best_match_idx = idx


            # Display recognition result
            if min_distance < 0.5:  # Adjust threshold as needed
                label = f"Match: {os.listdir(reference_folder)[best_match_idx]}"
            else:
                label = "Unknown"

            # Draw bounding box around face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Face Detection and Recognition", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()