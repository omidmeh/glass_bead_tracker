import numpy as np
import cv2
from os import chdir
import convenience as con
from tracker import Tracker

# Set Path
chdir(r'C:\Users\omidm\OneDrive\Uni\_HONERS PROJ\pyCode')
VIDEO_PATH = "..\\Videos\\Cut.mp4"

params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200
# Filter by Area.
params.filterByArea = True
params.minArea = 50
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01
# Filter by Color
params.filterByColor = False
params.minDistBetweenBlobs = 5

params.minDistBetweenBlobs = 30;


line_position = 500

count = 0
keypoints = None
firstFrame = None

# Video Processors
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
detector = cv2.SimpleBlobDetector_create(params)
kernel = np.ones((5, 5), np.uint8)

tracker = Tracker(line_pos=500,
                  track_vertically=True,
                  min_size=50,
                  threshold=35,
                  remove_if_missing_for=10)


def contour_center(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx, cy)


def draw_bounding_box(c, frame):
    # compute the bounding box for the contour, draw it on the frame, and update the text
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 2)
    cv2.circle(frame, contour_center(c), radius=0, thickness=5, color=(0, 255, 0))


try:
    cap.release()
    print("Released Camera")
except:
    print ("Camera was free")


# Load Video
cap = cv2.VideoCapture(VIDEO_PATH)
frame_counter = 0
blob_count = 0
xx = 1
while True:
    ret, frame = cap.read()
    if not ret: break

    if xx == 0:
        xx = 1;
        continue
    else: xx = 0

    frame_counter += 1
    frame = con.resize(frame, width=750)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 2)
    fgmask = fgbg.apply(gray)

    if firstFrame is None:
        firstFrame = gray
        continue

    # frameDelta = cv2.absdiff(firstFrame, gray)
    # thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    keypoints = detector.detect(opening)
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Draw a diagonal blue line with thickness of 5 px
    cv2.putText(frame, str(len(keypoints)), (175, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255, 255, 255))
    cv2.putText(im_with_keypoints, "Frame %d" % frame_counter, (600, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255, 255, 255))
    cv2.putText(im_with_keypoints, "keypts %d" % len(keypoints), (600, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255, 255, 255))

    tracker.process_keypoints(keypoints)
    tracker.draw_live_ids(im_with_keypoints)
    tracker.draw_predictions(im_with_keypoints)

    if blob_count != tracker.counted_blobs:
        cv2.line(im_with_keypoints, (200, 300), (300, 300), (150, 255, 255), 2)
        blob_count = tracker.counted_blobs
    else:
        cv2.line(im_with_keypoints, (200, 300), (300, 300), (150, 150, 255), 2)

    cv2.putText(im_with_keypoints, "Count: %s" % tracker.counted_blobs, (120, 295), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255, 255, 255))
    cv2.putText(im_with_keypoints, "Live %s" % tracker.live_blobs, (600, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255, 255, 255))

    # Display results
    # cv2.imshow('gray', gray)
    # cv2.imshow('fgmask', fgmask)
    # cv2.imshow('opening', opening)

    cv2.imshow('img + kpt',im_with_keypoints)
    # cv2.imshow('frame', frame)


    # Quit on 'Esc'
    k = cv2.waitKey(30) & 0xff
    # k = cv2.waitKey(-1) & 0xff
    if k == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

