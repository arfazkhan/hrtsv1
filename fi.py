import cv2
import face_recognition
import time
import os
from collections import namedtuple
import webbrowser

# Named tuple for face data
FaceData = namedtuple('FaceData', ['encoding', 'location'])

# Function to load reference images and their encodings
def load_reference_images(folder):
    reference_images = []
    reference_encodings = []
    
    for file in os.listdir(folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            image_path = os.path.join(folder, file)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            reference_images.append(image)
            reference_encodings.append(encoding)
    
    return reference_images, reference_encodings

def open_patient_profile(patient_id):
    profile_url = f"http://127.0.0.1:5000/patient/{patient_id}/"
    webbrowser.open(profile_url)

def find_patient_id(face_encoding, reference_encodings, prefix):
    for i, reference_encoding in enumerate(reference_encodings):
        if face_recognition.compare_faces([reference_encoding], face_encoding, tolerance=0.6)[0]:
            return f"{prefix}{i+1:03d}"
    return None

def recognize_face():
    # Load reference images and encodings from the folder
    reference_folder = "static/images"
    _, reference_encodings = load_reference_images(reference_folder)
    
    # Initialize webcam
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW helps in resolving headless issue

    start_time = time.time()
    detection_start_time = None
    patient_profile_updated = False

    prefix = 'HRTS2024'

    while not patient_profile_updated:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if detection_start_time is None:
            detection_start_time = time.time()

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        
        if face_locations:
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            # Create face data for each face
            faces = [FaceData(encoding, location) for encoding, location in zip(face_encodings, face_locations)]

            # Find the patient ID based on the face encoding
            for face in faces:
                patient_id = find_patient_id(face.encoding, reference_encodings, prefix)
                if patient_id:
                    open_patient_profile(patient_id)
                    patient_profile_updated = True
                    break

            # Save the detected face image
            top, right, bottom, left = faces[0].location  # Assuming only one face is detected
            face_image = frame[top:bottom, left:right]
            cv2.imwrite(f"detected_faces/detected_face_{int(time.time())}.jpg", face_image)

        # Break the loop if it's been more than 30 seconds or 'q' is pressed
        if time.time() - detection_start_time > 30 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()

    # Release webcam
    video_capture.release()

if __name__ == '__main__':
    recognize_face()
