import cv2
import numpy as np

# Load the reference image
reference_image = cv2.imread('1.jpg')
reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Create a face detector object using Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Extract features from the reference image (e.g., using Eigenfaces)
# Here, we'll just resize the reference image and convert it to grayscale
reference_gray_resized = cv2.resize(reference_gray, (100, 100))

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract features from the detected face region (e.g., resize and convert to grayscale)
        detected_face = gray[y:y+h, x:x+w]
        detected_face_resized = cv2.resize(detected_face, (100, 100))

        # Compare the features of the detected face with the reference features
        # Here, you would typically use a comparison method such as Euclidean distance or cosine similarity
        # For simplicity, let's just compare pixel-wise similarity
        similarity_score = np.mean(np.abs(detected_face_resized - reference_gray_resized))

        # Define a threshold for recognition
        recognition_threshold = 40

        # If the similarity score is below the threshold, recognize the person
        if similarity_score < recognition_threshold:
            cv2.putText(frame, 'Recognized', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection and Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
