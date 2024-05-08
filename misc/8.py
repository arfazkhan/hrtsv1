import cv2
import face_recognition

# Load reference image
reference_image = face_recognition.load_image_file("1.jpg")
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the image from BGR color to RGB color
    rgb_frame = frame[:, :, ::-1]

    # Find all face locations in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)

    for (top, right, bottom, left) in face_locations:
        # Crop the face region from the frame
        face_image = frame[top:bottom, left:right]

        # Convert the cropped face region from BGR to RGB
        rgb_face_image = face_image[:, :, ::-1]

        # Compute face encoding for the cropped face region
        face_encoding = face_recognition.face_encodings(rgb_face_image)

        # Compare the face encoding of the current frame with the reference face encoding
        if len(face_encoding) > 0:
            match = face_recognition.compare_faces([reference_encoding], face_encoding[0])

            # Display the result
            name = "Unknown"
            if match[0]:
                name = "Recognized"
                # Do something if recognized

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with the name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
