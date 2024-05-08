import cv2

# Load the reference image
reference_image = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)

# Detect faces in the reference image
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
reference_faces = face_cascade.detectMultiScale(reference_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Calculate histogram for the reference face
if len(reference_faces) > 0:
    x, y, w, h = reference_faces[0]
    reference_face = reference_image[y:y+h, x:x+w]
    reference_hist = cv2.calcHist([reference_face], [0], None, [256], [0, 256])
    cv2.normalize(reference_hist, reference_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

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
            # Calculate histogram for the detected face
            face_roi = gray_frame[y:y+h, x:x+w]
            face_hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
            cv2.normalize(face_hist, face_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            # Compare histograms
            correlation = cv2.compareHist(reference_hist, face_hist, cv2.HISTCMP_CORREL)
            
            # If correlation is high, consider it a match
            if correlation > 0.7:
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
