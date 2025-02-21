import cv2, time

video = cv2.VideoCapture(0)
time.sleep(1)

while True:
    check, frame = video.read()
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey_frame_gauss = cv2.GaussianBlur(grey_frame, (21,21), 0)

    cv2.imshow("My Video", grey_frame_gauss)

    key = cv2.waitKey()

    if key == ord("q"):
        break

video.release()
