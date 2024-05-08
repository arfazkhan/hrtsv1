import concurrent.futures
import cv2
import face_recognition
import time
import os
from collections import namedtuple


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

# Initialize webcam
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW helps in resolving headless issue

start_time = time.time()
detection_start_time = None
unknown_face_detected = True

# Load reference images and encodings from the folder
reference_folder = "reference_images"
reference_images, reference_encodings = load_reference_images(reference_folder)

while unknown_face_detected:
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

        # Function to compare face encodings
        def compare_encoding(face):
            for i, reference_encoding in enumerate(reference_encodings):
                if face_recognition.compare_faces([reference_encoding], face.encoding)[0]:
                    return i + 1
            return None

        # Use multithreading to perform face recognition
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(compare_encoding, faces)

        for face, result in zip(faces, results):
            if result:
                name = f"Reference Person {result}"
                html_file = f"{result}.html"  # Replace with the actual filename
                os.system(f'start {html_file}')  # Open in a new window
                unknown_face_detected = False
                break
        else:
            name = "Unknown"

        # Save the detected face image
        top, right, bottom, left = face.location
        face_image = frame[top:bottom, left:right]
        if unknown_face_detected:
            cv2.imwrite(f"detected_faces/detected_face_{int(time.time())}.jpg", face_image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()

# Release webcam
video_capture.release()
