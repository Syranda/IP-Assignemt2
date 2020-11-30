import numpy as np
import cv2
import matplotlib.pyplot as plt
print(cv2.__version__)

img = cv2.imread('IMG_3517.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp, desc = sift.detectAndCompute(gray, None)

img = cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpeg', img)


detect = cv2.imread('detect.jpeg')
gray_detect = cv2.cvtColor(detect, cv2.COLOR_BGR2GRAY)
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

MIN_MATCH_COUNT = 4

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
    text = cv2.putText(detect, 'Raucher', (text_transform[0][0][0], text_transform[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    detect = cv2.polylines(text, [np.int32(dst)],True,255,3, cv2.LINE_AA)
    cv2.imwrite('test2.jpeg', detect)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None