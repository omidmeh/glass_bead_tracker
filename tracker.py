import numpy as np
import cv2
from Blob import Blob
from math import sqrt


class Tracker:
    # Local variables
    _blobs = list()                 # type: list([Blob])
    _counted_blobs = 0              # type: int
    _track_vertically = None        # type: bool
    _last_used_ID = 0               # type: int
    # Parameters
    _line_position = None           # type: int
    _min_blob_size = 0              # type: int
    _blob_distance_threshold = 0    # type: int
    _remove_if_missing_for = 0      # type: int
    _debug_print = None             # type: bool

    def __init__(self, line_pos, track_vertically=True, min_size=50, threshold=25, remove_if_missing_for=10):
        # Initialization
        self._blobs = list()
        self._counted_blobs = 0
        self._track_vertically = track_vertically
        # Parameters
        self._line_position = line_pos
        self._min_blob_size = min_size
        self._blob_distance_threshold = threshold
        self._remove_if_missing_for = remove_if_missing_for
        self._debug_print = False

    def set_debug_print(self, debug_print_bool):
        self._debug_print = debug_print_bool

    def _new_blob(self):
        self._last_used_ID += 1
        return Blob(3, self._last_used_ID)

    def process_keypoints(self, keypoints):
        self._blobs = [blob for blob in self._blobs if blob.is_missing_for() < self._remove_if_missing_for]

        if len(self._blobs) == 0:
            for kp in keypoints:
                b = self._new_blob()
                b.register_new_loc(np.array(kp.pt))
                self._blobs.append(b)
        else:
            unmatched_registered_blobs = dict.fromkeys(self._blobs)
            # unmatched_registered_blobs = dict()

            for kp in keypoints:
                # for blob in self._blobs:
                for blob in unmatched_registered_blobs:
                    pt1 = kp.pt
                    pt2 = blob.predict_next_coordinate()
                    unmatched_registered_blobs[blob] = sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)

                if len(unmatched_registered_blobs) == 0:
                    b = self._new_blob()
                    b.register_new_loc(np.array(kp.pt))
                    self._blobs.append(b)
                    if self._debug_print:
                        print("Added blob*")

                else:
                    mm = min(unmatched_registered_blobs, key=unmatched_registered_blobs.get)

                    if self._debug_print:
                        print("dist: {:6.2f} | new: {} | predict: {}".format(unmatched_registered_blobs[mm],
                                                                             np.around(np.array(kp.pt), 2),
                                                                             np.around(mm.predict_next_coordinate(), 2))),
                    if unmatched_registered_blobs[mm] < self._blob_distance_threshold:
                        mm.register_new_loc(np.array(kp.pt))
                        unmatched_registered_blobs.pop(mm, 0)
                        if mm.crossed_h_line(300):
                            self._counted_blobs += 1
                        if self._debug_print:
                            print("Matched blob %d" % mm.id)
                    else:
                        b = self._new_blob()
                        b.register_new_loc(np.array(kp.pt))
                        self._blobs.append(b)
                        if self._debug_print:
                            print("Added blob")

            for unmatched in unmatched_registered_blobs:
                unmatched.mark_missing_in_this_frame()
                # print "unmatched: %d" % len(unmatched_registered_blobs)

    def draw_live_ids(self, mat, draw_missing=False):

        for blob in self._blobs:
            if not draw_missing:
                if blob.is_missing_for() == 0:
                    cv2.putText(mat, "%d" % blob.id,
                                (int(blob.last_known_coordinate[0]), int(blob.last_known_coordinate[1])),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
            else:
                cv2.putText(mat, "%d" % blob.id,
                            (int(blob.last_known_coordinate[0]), int(blob.last_known_coordinate[1])),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))

    def draw_predictions(self, mat, draw_missing=False):
        for blob in self._blobs:
            if not draw_missing:
                if blob.is_missing_for() == 0:
                    predict = (int(blob.predict_next_coordinate()[0]), int(blob.predict_next_coordinate()[1]))
                    cv2.circle(mat, predict, 2, (255, 0, 0), 3)
            else:
                predict = (int(blob.predict_next_coordinate()[0]), int(blob.predict_next_coordinate()[1]))
                cv2.circle(mat, predict, 2, (255, 0, 0), 3)

    @property
    def live_blobs(self):
        return len(self._blobs)

    @property
    def counted_blobs(self):
        return self._counted_blobs
