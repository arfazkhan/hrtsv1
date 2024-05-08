import cv2
import dlib
from mtcnn import MTCNN

# Load the reference image
reference_image = cv2.imread('1.jpg')
reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Initialize MTCNN and Dlib
mtcnn = MTCNN()
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Compute face embeddings for the reference image
faces_reference = face_detector(reference_gray)
face_descriptors_reference = [face_recognizer.compute_face_descriptor(reference_gray, shape_predictor(reference_image, face)) for face in faces_reference]

# Capture live feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame using MTCNN
    faces = mtcnn.detect_faces(frame)

    # Convert frame to grayscale for Dlib
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Loop through detected faces
    for result in faces:
        x, y, w, h = result['box']
        face = frame[y:y+h, x:x+w]

        # Compute face embedding using Dlib
        face_dlib = cv2.resize(face, (150, 150))
        face_dlib_gray = cv2.cvtColor(face_dlib, cv2.COLOR_BGR2GRAY)
        face_descriptor = face_recognizer.compute_face_descriptor(face_dlib_gray)

        # Compare face embedding with reference image
        match = False
        for ref_descriptor in face_descriptors_reference:
            distance = dlib.euclidean_distance(ref_descriptor, face_descriptor)
            if distance < 0.6:  # You can adjust this threshold as needed
                match = True
                break

        # Display result
        if match:
            cv2.putText(frame, "Matched", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
