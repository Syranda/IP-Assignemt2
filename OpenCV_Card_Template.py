'''
This file contains the template for the card recognition exercise

Follow the instructions given in the exercise sheet and complete the missing code sections.
You'll find excellent information and examples within the OpenCV documentation

'''


import cv2
import numpy as np

# defines minimum matches necessary to detect card
MIN_MATCH_COUNT = ...

# make sure OpenCV version is 3.4.2 (SURF/SIFT are inluded in the contrib distribution, otherwise OpenCV has to be
# compiled from scratch
print(cv2.__version__)

# create video object: argument can be either the device index (usually 0) or the name of a video file.
cap = cv2.VideoCapture(0)

# FLANN matcher and its parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(...)

# or alternatively brute force matcher
bf = cv2.BFMatcher(...)


modelimg1= cv2.imread(...)

# create sift detector object, compute model features and draw them onto the model(s)
sift = cv2.xfeatures2d.SIFT_create(...)
...
...

# just an idea for building up a model database. One can easily add multiple cards or another objects to be detected)
model_db=[]
model_db.append([keypoints, descriptors, 'item', modelimg1])


cv2.imshow('Model Features', ...)
cv2.waitKey(0)

while(True):
    # capture frame-by-frame: cap.read() returns a bool (True/False). Attention: on some systems the first captured
    # frame migth be empty. Catch it by checking the 'ret' for being true.
    ret, frame = cap.read()



    ...
    ...
    ...


    for entry in model_db:
        # match descriptors from frame
        matches = ...
        ...


        cv2.imshow("Card Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Before exiting the program, release the capture
cap.release()
cv2.destroyAllWindows()



# in order to write a captured video
#
# define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
#
# #grab frame within a while loop and do
#
# if ret==True:
#     frame = cv2.flip(frame,0)
#
#     # write the flipped frame
#     out.write(frame)
#
#
# # finally release writer object
# out.release()