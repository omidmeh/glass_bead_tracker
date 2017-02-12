import numpy


class Blob:
    """ One per blob"""
    _ID = None
    _last_xy_points = None         # type: list([numpy.ndarray])
    _missing_frame_count = None    # type: int
    _history_length = None         # type: int

    def __init__(self, history=3, ID=0):
        """ Initializes the blob object.

        :param history: number of old positions to keep for calculating object's velocity (default 3)
        """
        self._ID = ID
        self._last_xy_points = list()
        self._missing_frame_count = 0
        self._history_length = history

    def register_new_loc(self, new_coordinate):
        """ Register the new position of the blob.

        :type new_coordinate: numpy.ndarray
        :param new_coordinate: new position of the blob in (x,y) format
        """
        assert (len(self._last_xy_points) <= self._history_length)

        if len(self._last_xy_points) == self._history_length:
            self._last_xy_points.pop(0)

        self._last_xy_points.append(new_coordinate)
        self._missing_frame_count = 0

    def print_positions(self):
        """ Prints the last registered coordinates of the blob (up to _history_length many of them.)"""

        print ("Positions for blob id %03d: ", self._ID)
        for pos in self._last_xy_points:
            print ("%s " % (pos,)),
        print("")   # New line

    def predict_next_coordinate(self):
        """ Predicts next coordinate of the blob

        If there are no past coordinates registered, returns (0, 0)
        If there is one past coordinate  registered, returns those coordinates
        If there are more than one past coordinates registered:
          Evaluates average vx, and vy, and predicts the next coordinate using x = vt + x0

        :return: next position of the blob in (x, y) format
        """
        if len(self._last_xy_points) == 0:
            return 0, 0

        if len(self._last_xy_points) == 1:
            return self._last_xy_points[0]

        pt1 = self._last_xy_points[0]
        pt2 = self._last_xy_points[-1]

        dx = (pt2[0]*1.0 - pt1[0]) / (len(self._last_xy_points) - 1)
        dy = (pt2[1]*1.0 - pt1[1]) / (len(self._last_xy_points) - 1)

        x2 = pt2[0] + (dx * (1 + self._missing_frame_count))
        y2 = pt2[1] + (dy * (1 + self._missing_frame_count))

        return x2, y2

    def mark_missing_in_this_frame(self):
        """ Marks this blob as missing in this frame. """
        self._missing_frame_count += 1

    def is_missing_for(self):
        """ Returns how many frames the blob was missing """

        return self._missing_frame_count

    def crossed_h_line(self, line_y_position):
        """ Returns if in the last frame this blob crossed a horizontal line at y = line_y_position"""

        assert (self._history_length > 0)

        m1 = self._last_xy_points[-1][1] - line_y_position
        m2 = self._last_xy_points[-2][1] - line_y_position

        return (m1 * m2) < 0

    def crossed_v_line(self, line_x_position):
        """ Returns if in the last frame this blob crossed a vertical line at x = line_x_position"""

        assert (self._history_length > 1)

        m1 = self._last_xy_points[-1][0] - line_x_position
        m2 = self._last_xy_points[-2][0] - line_x_position

        return (m1 * m2) < 0

    @property
    def id(self):
        return self._ID

    @property
    def last_known_coordinate(self):
        assert len(self._last_xy_points[-1]) > 0
        return self._last_xy_points[-1]
