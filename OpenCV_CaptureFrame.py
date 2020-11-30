''' Salzburg University of Applied Sciences

Python program to capture a frame from a camera's video stream.
    - press "c" to capture frame
    - press "q" to quit program

Author: Werner Pomwenger, 2018.
'''

import cv2
import matplotlib
import numpy as np

print(cv2.__version__)

img = cv2.imread('IMG_3574.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp, desc = sift.detectAndCompute(gray, None)

img = cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpeg', img)

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
    # cv2.imshow('Frame', frame)
    #To quit the live stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #To capture a frame
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.waitKey(0)
        cv2.imwrite('modelimg.png', frame)
        print('Frame captured...')

    gray_detect = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect_kp, detect_desc = sift.detectAndCompute(gray_detect, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc, detect_desc, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 200
    print(len(good))

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ detect_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w,d = img.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        text_transform = cv2.perspectiveTransform(np.float32([0, 0]).reshape(-1,1,2), M)
        print((text_transform[0][0][0], text_transform[0][0][1]))
        text = cv2.putText(frame, 'Raucher', (text_transform[0][0][0], text_transform[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        detect = cv2.polylines(text, [np.int32(dst)],True,255,3, cv2.LINE_AA)
        cv2.imshow('Frame', detect)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

# before exiting the program, release the capture
cap.release()
cv2.destroyAllWindows()


