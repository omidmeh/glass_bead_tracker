from os import chdir

import numpy as np
import cv2
import convenience as con
from tracker import Tracker
import configparser
from os import chdir

chdir('./input-output')
Config = configparser.ConfigParser()
Config.read('./config.ini')

# CONFIGURATION #
# Set Path
video_in =  Config.get('video', 'input_video')
video_out = Config.get('video', 'output_video')

# Tracker Initialization
tracker = Tracker(line_pos          =Config.getfloat('tracking',     'line_position'),
                  track_vertically  =Config.getboolean('tracking', 'track_vertically'),
                  min_size          =Config.getfloat('tracking', 'min_size'),
                  threshold         =Config.getfloat('tracking', 'threshold'),
                  remove_if_missing_for=Config.getint('tracking', 'remove_if_missing_for'))

# Blob Detector Setup
params = cv2.SimpleBlobDetector_Params()
params.minThreshold =           Config.getfloat('blob_detector',    'minThreshold')
params.maxThreshold =           Config.getfloat('blob_detector',    'maxThreshold')
params.filterByArea =           Config.getboolean('blob_detector',  'filterByArea')
params.minArea =                Config.getfloat('blob_detector',    'minArea')
params.filterByCircularity =    Config.getboolean('blob_detector',  'filterByCircularity')
params.minCircularity =         Config.getfloat('blob_detector',    'minCircularity')
params.filterByConvexity =      Config.getboolean('blob_detector',  'filterByConvexity')
params.minConvexity =           Config.getfloat('blob_detector',    'minConvexity')
params.filterByInertia =        Config.getboolean('blob_detector',  'filterByInertia')
params.minInertiaRatio =        Config.getfloat('blob_detector',    'minInertiaRatio')
params.filterByColor =          Config.getboolean('blob_detector',  'filterByColor')
params.minDistBetweenBlobs =    Config.getfloat('blob_detector',    'minDistBetweenBlobs')

# Video Processors
kernel = np.ones((5, 5), np.uint8)
detector = cv2.SimpleBlobDetector_create(params)
fgbg = cv2.createBackgroundSubtractorMOG2(history       =Config.getint('bg_subtract', 'history'),
                                          varThreshold  =Config.getint('bg_subtract', 'varThreshold'),
                                          detectShadows =Config.getboolean('bg_subtract', 'detectShadow'))

# frame Processing parms
line_position = Config.getfloat('tracking', 'line_position')
skip_every_other_frame = Config.getboolean('video', 'skip_every_other_frame')
resize_w = Config.getint('video', 'resized_width')
show_debug_img = Config.getboolean('debug', 'show_intermediate_images')
show_more_info = Config.getboolean('debug', 'show_more_info')
print_debug_info = Config.getboolean('debug', 'print_debug_info')


try:
    cap.release()
    print("Released Camera")
except:
    print ("Camera was free")

count = 0
keypoints = None
firstFrame = None
frame_counter = 0
blob_count = 0

# Load Video
cap = cv2.VideoCapture(video_in)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
resize_h = int(h * (resize_w / float(w)))
out = cv2.VideoWriter(video_out, -1, 20.0, (resize_w, resize_h))


while True:
    ret, frame = cap.read()
    if skip_every_other_frame:
        ret, frame = cap.read()
    if not ret: break

    frame_counter += 1
    frame = con.resize(frame, width=resize_w)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 2)
    fgmask = fgbg.apply(gray)

    if firstFrame is None:
        firstFrame = gray
        continue

    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    keypoints = detector.detect(opening)
    tracker.process_keypoints(keypoints)

    # Write info on the image
    annotated_frame = frame
    cv2.putText(annotated_frame, "Count: %s" % tracker.counted_blobs, (120, 295), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255, 255, 255))

    # Draw a diagonal blue line with thickness of 5 px
    if blob_count != tracker.counted_blobs:
        cv2.line(annotated_frame, (200, 300), (300, 300), (150, 255, 255), 2)
        blob_count = tracker.counted_blobs
    else:
        cv2.line(annotated_frame, (200, 300), (300, 300), (150, 150, 255), 2)

    if show_more_info:
        annotated_frame = cv2.drawKeypoints(annotated_frame, keypoints, np.array([]), (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        tracker.draw_live_ids(annotated_frame)
        tracker.draw_predictions(annotated_frame)
        cv2.putText(annotated_frame, "Frame %d" % frame_counter, (600, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255, 255, 255))
        cv2.putText(annotated_frame, "keypts %d" % len(keypoints), (600, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255, 255, 255))
        cv2.putText(annotated_frame, "Live %s" % tracker.live_blobs, (600, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255, 255, 255))


    # Display results
    if show_debug_img:
        cv2.imshow('gray', gray)
        cv2.imshow('fgmask', fgmask)
        cv2.imshow('opening', opening)

    cv2.imshow('result', annotated_frame)
    if out is not None:
        out.write(annotated_frame)

    # Quit on 'Esc'
    k = cv2.waitKey(30) & 0xff
    # k = cv2.waitKey(-1)
    if k == 27:
        break
    if k == ord('d'):
        show_debug_img = not show_debug_img
        cv2.destroyAllWindows()
    if k == ord('i'):
        show_more_info = not show_more_info
    if k == ord('p'):
        print_debug_info = not print_debug_info
        tracker.set_debug_print(print_debug_info)


# Release resources
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()

