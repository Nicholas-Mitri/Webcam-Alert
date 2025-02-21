import cv2, time

# Initialize video capture from webcam
video = cv2.VideoCapture(0)
time.sleep(1)

# Initialize first frame as None
first_frame = None
count = 1

while True:
    # Read frame from video
    check, frame = video.read()
    # Convert frame to grayscale
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    grey_frame_gauss = cv2.GaussianBlur(grey_frame, (21,21), 0)

    # Set first frame if not already set
    if not first_frame:
        first_frame = grey_frame_gauss

    # Calculate absolute difference between first frame and current frame
    delta_frame = cv2.absdiff(first_frame, grey_frame_gauss)

    # Apply threshold to get binary image
    thresh_frame = cv2.threshold(delta_frame, 45, 255, cv2.THRESH_BINARY)[1]

    # Dilate thresholded image to fill in holes
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Find contours in dilated frame
    contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process each contour
    for contour in contours:
        # Skip small contours
        if cv2.contourArea(contour)< 5000:
            continue
        # Get bounding rectangle coordinates
        x, y , w, h = cv2.boundingRect(contour)
        # Draw rectangle around detected object
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0))

    # Save frame as image if motion detected (contours found)
    if contours:
        cv2.imwrite(f"images/img_{count}.png", frame)
        count += 1

    # Display the frame
    cv2.imshow("My Video", frame)

    # Check for 'q' key press to quit
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release video capture
video.release()
