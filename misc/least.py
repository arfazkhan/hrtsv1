import cv2

# Load the reference image
reference_image = cv2.imread("1.jpg")
reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
reference_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
reference_faces = reference_face_cascade.detectMultiScale(reference_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = reference_face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If faces are found
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Compare the size of detected faces with the reference face
            if len(reference_faces) > 0:
                ref_x, ref_y, ref_w, ref_h = reference_faces[0]
                if abs((w * h) - (ref_w * ref_h)) < 1000:
                    # Draw a rectangle around the face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Face Detection and Recognition', frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
