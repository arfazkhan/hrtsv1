import cv2
import numpy as np

# Load the reference image
reference_image = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)

# Detect faces in the reference image
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
reference_faces = face_cascade.detectMultiScale(reference_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Initialize LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# If faces are found in the reference image
if len(reference_faces) > 0:
    x, y, w, h = reference_faces[0]
    reference_face = reference_image[y:y+h, x:x+w]
    face_recognizer.train([reference_face], np.array([0]))

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If faces are found
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Recognize face using LBPH
            face_roi = gray_frame[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(face_roi)
            
            # If confidence is low, consider it a match
            if confidence < 100:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, 'Match', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Display the resulting frame
    cv2.imshow('Face Detection and Recognition', frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
