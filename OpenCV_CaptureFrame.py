''' Salzburg University of Applied Sciences

Python program to capture a frame from a camera's video stream.
    - press "c" to capture frame
    - press "q" to quit program

Author: Werner Pomwenger, 2018.
'''

import cv2

print(cv2.__version__)

# create video object: argument is the device index (usually 0 if only one video device is present. also a video file
# can be provided
cap = cv2.VideoCapture(0)

while(True):

    # capture video stream frame-by-frame: cap.read() returns a bool (True/False) and a frame
    ret, frame = cap.read()

    if ret == False:
        print('No data in frame.')
        continue

    # display the frame, quit video stream by pressing 'q'
    cv2.imshow('Frame', frame)
    #To quit the live stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #To capture a frame
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.waitKey(0)
        cv2.imwrite('modelimg.png', frame)
        print('Frame captured...')

# before exiting the program, release the capture
cap.release()
cv2.destroyAllWindows()


