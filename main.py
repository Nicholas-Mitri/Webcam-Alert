import cv2, time

video = cv2.VideoCapture(0)
time.sleep(1)

first_frame = None
while True:
    check, frame = video.read()
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey_frame_gauss = cv2.GaussianBlur(grey_frame, (21,21), 0)


    if not first_frame:
        first_frame = grey_frame_gauss

    delta_frame = cv2.absdiff(first_frame, grey_frame_gauss)

    thresh_frame = cv2.threshold(delta_frame, 45, 255, cv2.THRESH_BINARY)[1]
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)

    contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour)< 5000:
            continue
        x, y , w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0))

    cv2.imshow("My Video", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

video.release()
