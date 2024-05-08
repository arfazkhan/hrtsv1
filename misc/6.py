import dlib
import cv2
import numpy as np

# Load the pre-trained face detector
detector = dlib.get_frontal_face_detector()

# Load the pre-trained face recognition model
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load the reference image for recognition
reference_image = dlib.load_rgb_image("1.jpg")
reference_faces = detector(reference_image)

# Extract face descriptor from the reference image
reference_face_descriptors = []
for face in reference_faces:
    landmarks = shape_predictor(reference_image, face)
    reference_face_descriptors.append(face_recognizer.compute_face_descriptor(reference_image, landmarks))

# Function to recognize faces in live feed
def recognize_faces_live():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert the frame to RGB for dlib
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        faces = detector(rgb_frame)

        for face in faces:
            # Upscale detected face coordinates
            face = dlib.rectangle(
                int(face.left() * 2),
                int(face.top() * 2),
                int(face.right() * 2),
                int(face.bottom() * 2)
            )

            # Get the face landmarks
            landmarks = shape_predictor(frame, face)

            # Compute the face descriptor
            face_descriptor = face_recognizer.compute_face_descriptor(frame, landmarks)

            # Compare the face descriptor with the reference face descriptors
            distances = [np.linalg.norm(np.array(face_descriptor) - np.array(reference_descriptor)) for reference_descriptor in reference_face_descriptors]

            # Recognize the face if the distance is below a certain threshold
            if min(distances) < 0.6:
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                cv2.putText(frame, "Recognized", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the function to recognize faces in live feed
recognize_faces_live()
