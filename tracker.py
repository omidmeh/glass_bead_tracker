import numpy as np
import cv2
from Blob import Blob


class Tracker():
    _line_position = None       #type: int
    _track_vertically = None    #type: bool
    _blobs = list()             #type: List[Blob]
    _min_cnt_size = 0           #type: int
    _threshold = 25             #type: int
    _crossed = 0                #type: int

    def __init__(self, line_pos, track_vertically=True, min_size=50):
        self._line_position = line_pos
        self.track_vertically = track_vertically
        self._min_cnt_size = min_size

    def process_keypoints(self, keypoints):
        current_pos = np.array([])
        # self._blobs = filter(lambda x: x.is_missing_for < 10, self._blobs)
        self._blobs = [blob for blob in self._blobs if blob.is_missing_for() < 10]

        if len(self._blobs) == 0:
            for kp in keypoints:
                b = Blob()
                b.register_new_loc(np.array(kp.pt))
                self._blobs.append(b)
        else:
            distance_to_blobs = dict(zip(self._blobs, [0 for x in self._blobs]))

            for kp in keypoints:
                for blob in distance_to_blobs:
                    distance_to_blobs[blob] = np.linalg.norm(np.array(kp.pt) - blob.predict_next_coordinate())

                if len(distance_to_blobs) == 0:
                    b = Blob()
                    b.register_new_loc(np.array(kp.pt))
                    self._blobs.append(b)
                    print("Added blob*")

                else:
                    mm = min(distance_to_blobs, key=distance_to_blobs.get)

                    print("dist: {:6.2f} | new: {} | predict: {}".format(distance_to_blobs[mm],
                                                                         np.around(np.array(kp.pt), 2),
                                                                         np.around(mm.predict_next_coordinate(), 2))),
                    if distance_to_blobs[mm] < self._threshold:
                        mm.register_new_loc(np.array(kp.pt))
                        distance_to_blobs.pop(mm)
                        if mm.crossed_h_line (300):
                            self._crossed += 1
                        print("Matched blob")
                    else:
                        b = Blob()
                        b.register_new_loc(np.array(kp.pt))
                        self._blobs.append(b)
                        print("Added blob")

            for unmatched in distance_to_blobs:
                unmatched.mark_missing_in_this_frame()
                # print "unmatched: %d" % len(distance_to_blobs)

            # self._blobs = [b for b in self._blobs if b.is_missing_for < 10]










        pass

    def draw_current_keypoints(self, mat):
        pass

    def draw_predicted_centers(self, mat):
        pass