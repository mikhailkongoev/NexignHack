from __future__ import print_function

import math
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from linear_assignment_ import linear_assignment

from kalmanfilter import KalmanFilter

OUTSIDE_CONTOUR = -1
INSIDE_CONTOUR = 1
BORDER = 0

SOFT_IOU_ADD = 0.25

"""
This algorithm based on https://github.com/abewley/sort
"""
import json


class OutputImgs:
    def __init__(self, faces, imgs_from_camera, timestamps):
        self.faces = faces
        self.imgs = imgs_from_camera
        self.vectors = []
        self.timestamps = timestamps

    def add_vector(self, vector):
        self.vectors.append(vector)

    def not_empty(self):
        return len(self.vectors) > 0


class Box:
    def __init__(self, track_id, frame, phantom, centered=None, bbox=None,
                 left=None, right=None, top=None, bottom=None, current_class=None):
        if centered is not None:
            w = centered[2]
            h = centered[3]
            left = centered[0] - w / 2.
            right = centered[0] + w / 2.
            top = centered[1] - h / 2.
            bottom = centered[1] + h / 2.

        if bbox is not None:
            left = bbox[0]
            top = bbox[1]
            right = bbox[2]
            bottom = bbox[3]

        self.track_id = track_id
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.frame = frame
        self.phantom = phantom
        self.frame_img = None
        self.mid_p_x = (self.left + self.right) / 2
        self.mid_p_y = (self.top + self.bottom) / 2
        # if frame_img is not None and not self.phantom:
        #     self.frame_img = frame_img[self.top:self.bottom, self.left:self.right]
        self.current_class = current_class

    def area(self):
        return (self.bottom - self.top) * (self.right - self.left)

    def to_centered(self):
        w = self.right - self.left
        h = self.bottom - self.top
        x = self.left + w / 2.
        y = self.top + h / 2.
        return np.array([x, y, w, h]).reshape((4, 1))

    def to_bbox(self):
        return np.array([self.left, self.top, self.right, self.bottom]).reshape((4, 1))


class Track:
    def __init__(self, track_id, box, frame, vx, vy, ax, ay, pawnshop):
        self.track_id = track_id
        self.boxes = [box]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, vx, 0, 0, 0],
             [0, 1, 0, 0, 0, vy, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, ax, 0],
             [0, 0, 0, 0, 0, 1, 0, ay],
             [0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = box.to_centered()
        self.kf.predict()
        self.last_frame = box.frame
        self.real_boxes = 1
        self.phantom_boxes = 0
        self.max_phantom_boxes = 0
        self.start_imgs = deque(maxlen=100)
        self.end_imgs = deque(maxlen=100)

        self.pawnshop = pawnshop

    def add_img(self, frame, box):
        timestamp = datetime.now()
        # timestamp = timestamp.strftime('%Y/%m/%d-%H:%M:%S')
        if type(box.top) != int:
            if len(self.start_imgs) == self.start_imgs.maxlen:
                self.end_imgs.append(
                    (
                        frame,
                        frame[int(box.top[0]):int(box.bottom[0]), int(box.left[0]):int(box.right[0])],
                        timestamp
                    )
                )
            else:
                self.start_imgs.append(
                    (
                        frame,
                        frame[int(box.top[0]):int(box.bottom[0]), int(box.left[0]):int(box.right[0])],
                        timestamp
                    )
                )

        else:
            if len(self.start_imgs) == self.start_imgs.maxlen:
                self.end_imgs.append(
                    (
                        frame,
                        frame[box.top:box.bottom, box.left:box.right],
                        timestamp
                    )
                )
            else:
                self.start_imgs.append(
                    (
                        frame,
                        frame[box.top:box.bottom, box.left:box.right],
                        timestamp
                    )
                )

    def update_phantom(self):
        new_box = self.get_prediction()
        self.last_frame += 1
        self.boxes.append(new_box)
        self.kf.update(self.kf.x[:4])
        self.kf.predict()
        self.phantom_boxes += 1
        if self.phantom_boxes > self.max_phantom_boxes:
            self.max_phantom_boxes = self.phantom_boxes

    def update_real(self, box, non_decrease, frame):
        if len(self.boxes) > 0:
            prev = self.boxes[-1]
            ratio = (box.right - box.left) * (box.bottom - box.top) / (
                    (prev.right - prev.left) * (prev.bottom - prev.top))
            if ratio < non_decrease:
                predicted = self.kf.x[:4]
                box = Box(self.track_id, self.last_frame, True, centered=predicted)

        self.boxes.append(box)
        self.real_boxes += 1
        self.phantom_boxes = 0
        self.add_img(frame, box)
        self.last_frame = box.frame
        self.kf.update(box.to_centered())
        self.kf.predict()

    def get_prediction(self):
        return Box(track_id=self.track_id, frame=self.last_frame, phantom=True, centered=self.kf.x)

    def get_max_phantoms(self):
        return self.max_phantom_boxes

    def get_last_real(self):
        i = len(self.boxes) - 1
        while i >= 0 and self.boxes[i].phantom:
            i -= 1

        if i == -1:
            return self.boxes[0]
        return self.boxes[i]

    def union_tracks(self, new_track_id):
        for i in range(len(self.boxes)):
            self.boxes[i].track_id = new_track_id

        self.track_id = new_track_id


class PhantomSortTrackerOnline:
    def __init__(self, min_x, min_y, max_x, max_y, detection_thresh, nms_thresh,
                 detections_count, track_creation_score, min_phantom_omit, max_phantom_omit,
                 phantom_coef, non_decrease, inc_size, vx, vy, ax, ay,
                 static_coef, soft_iou_k, area_threshold, min_box_size, track_by):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.detection_thresh = detection_thresh  # Threshold for detection probability
        self.nms_thresh = nms_thresh  # X
        self.detections_count = detections_count  # How much to be a person V
        self.track_creation_score = track_creation_score  # Follow person if detection_thresh < P < track_creation_score
        self.min_phantom_omit = min_phantom_omit  # About phantoms
        self.max_phantom_omit = max_phantom_omit  # About phantoms
        self.phantom_coef = phantom_coef  # About phantoms
        self.non_decrease = non_decrease  # Times
        self.inc_size = inc_size  # Times for far and close
        self.static_coef = static_coef  # Length of path V
        self.soft_iou_k = soft_iou_k  # X
        self.vx = vx  # Kalman filters velocity V
        self.vy = vy  # Kalman filters velocity V
        self.ax = ax  # Kalman filters acceleration V
        self.ay = ay  # Kalman filters acceleration V
        self.tracks = []
        self.next_id = 0
        self.objects = 0
        self.area_threshold = area_threshold  # V
        self.min_box_size = min_box_size
        self.track_by = track_by
        self.images = []

    @classmethod
    def from_json(cls, path_config):
        config = json.load(open(path_config, "r"))
        area_threshold = 0
        if "area_threshold" in config:
            area_threshold = config["area_threshold"]

        min_box_size = 50
        if "min_box_size" in config:
            min_box_size = config["min_box_size"]

        track_by = "full"
        if "track_by" in config:
            track_by = config["track_by"]

        return PhantomSortTrackerOnline(min_x=config["min_x"], min_y=config["min_y"], max_x=config["max_x"],
                                        max_y=config["max_y"],
                                        detection_thresh=config["detection-thresh"], nms_thresh=config["nms-thresh"],
                                        detections_count=config["detections-count"],
                                        track_creation_score=config["track-creation"],
                                        min_phantom_omit=config["min-phantom"], max_phantom_omit=config["max-phantom"],
                                        phantom_coef=config["phantom-coef"], non_decrease=config["non-decrease"],
                                        vx=config["vx"], vy=config["vy"], ax=config["ax"], ay=config["ay"],
                                        inc_size=config["inc-size"],
                                        static_coef=config["static-coef"], soft_iou_k=config["soft-iou-k"],
                                        area_threshold=area_threshold,
                                        min_box_size=min_box_size,
                                        track_by=track_by)

    def iou(self, a, b):
        w_a = a[2] - a[0]
        w_b = b[2] - b[0]
        h_a = a[3] - a[1]
        h_b = b[3] - b[1]
        ratio = (w_a * h_a) / (w_b * h_b)
        if 1 / self.inc_size <= ratio <= self.inc_size:
            x1 = np.maximum(a[0], b[0])
            y1 = np.maximum(a[1], b[1])
            x2 = np.minimum(a[2], b[2])
            y2 = np.minimum(a[3], b[3])
            w_c = np.maximum(0., x2 - x1)
            h_c = np.maximum(0., y2 - y1)
            s_c = w_c * h_c
            if s_c > 0:
                return SOFT_IOU_ADD + (1 - SOFT_IOU_ADD) * s_c / ((a[2] - a[0]) * (a[3] - a[1])
                                                                  + (b[2] - b[0]) * (b[3] - b[1]) - s_c)
            elif self.soft_iou_k > 0:
                dx = np.abs((a[0] + a[2]) - (b[0] + b[2])) / 2
                dy = np.abs((a[1] + a[3]) - (b[1] + b[3])) / 2
                soft_iou = SOFT_IOU_ADD * (1 - max(dx / (w_a + w_b), dy / (h_a + h_b)) / self.soft_iou_k)
                if soft_iou > 0:
                    return soft_iou
        return 0

    def get_point(self, box, track_by):
        if track_by == "legs":
            legs_x = (box.left + box.right) / 2
            legs_y = (box.top + 9 * box.bottom) / 10
            box_point = (legs_x, legs_y)
        elif track_by == "head":
            head_x = (box.left + box.right) / 2
            head_y = (box.top * 9 + box.bottom) / 10
            box_point = (head_x, head_y)
        else:
            point_x = (box.left + box.right) / 2
            point_y = (box.top + box.bottom) / 2
            box_point = (point_x, point_y)
        return box_point

    def update_all(self, detections, start_frame, current_image):
        colored_boxes = []
        frame = start_frame

        if len(detections) > 0:
            iou_matrix = np.zeros((len(detections), len(self.tracks) + len(detections)), dtype=np.float32)
            for d, det in enumerate(detections):
                for t, track in enumerate(self.tracks):
                    iou_matrix[d, t] = -self.iou(det, track.get_prediction().to_bbox())
                    if iou_matrix[d, t] < 0:
                        if not track.boxes[-1].phantom:
                            iou_matrix[d, t] -= 1.0
                if det[-1] < self.track_creation_score:
                    iou_matrix[d, len(self.tracks) + d] = +0.001
                else:
                    iou_matrix[d, len(self.tracks) + d] = -0.001
            matched_indices = linear_assignment(iou_matrix)
            old_length = len(self.tracks)
            for row in matched_indices:
                b = detections[row[0]]
                if row[1] >= old_length:
                    id = self.next_id
                    self.next_id += 1

                    new_track = Track(id, Box(id, frame, False, bbox=b), current_image,
                                      self.vx, self.vy, self.ax, self.ay, False)
                    self.tracks.append(new_track)
                elif iou_matrix[row[0], row[1]] < 0:
                    track = self.tracks[row[1]]

                    box = Box(track.track_id, frame, False, bbox=b)
                    track.update_real(box, self.non_decrease, current_image)

        active_tracks = []
        result = []
        for track in self.tracks:
            if track.last_frame < frame:
                track.update_phantom()
                # phantom_threshold = np.minimum(self.max_phantom_omit,
                #                                np.maximum(self.min_phantom_omit,
                #                                           self.phantom_coef * track.get_max_phantoms()))
                phantom_threshold = self.min_phantom_omit
                box = track.boxes[-1]
                if track.phantom_boxes > phantom_threshold or box.left > self.max_x \
                        or box.right < self.min_x \
                        or box.top < self.min_y \
                        or box.bottom > self.max_y:
                    print("remove track " + str(track.track_id))
                    result.append(track)
                else:
                    active_tracks.append(track)
            else:
                active_tracks.append(track)
            box = track.boxes[-1]
            colored_box = [int(box.left), int(box.bottom), int(box.right), int(box.top), int(track.track_id),
                           int(box.phantom)]
            colored_boxes.append(colored_box)
        self.tracks = active_tracks
        frame += 1

        return colored_boxes, result

    def get_img_from_tracks(self, tracks):
        result = []
        for track in tracks:
            faces = []
            target_imgs = []
            timestamps = []
            for i in range(len(track.end_imgs)):
                if i % 1 == 0:
                    faces.append(track.end_imgs[i][1])
                    # target_imgs.append(cv2.resize(track.end_imgs[i][0], (340, 200)))
                    target_imgs.append(track.end_imgs[i][0])
                    timestamps.append(track.end_imgs[i][2].strftime('%Y/%m/%d-%H:%M:%S'))
            if len(target_imgs) < 10:
                for i in range(len(track.start_imgs)):
                    if i % 1 == 0:
                        faces.append(track.start_imgs[i][1])
                        # target_imgs.append(cv2.resize(track.start_imgs[i][0], (340, 200)))
                        target_imgs.append(track.start_imgs[i][0])
                        timestamps.append(track.start_imgs[i][2].strftime('%Y/%m/%d-%H:%M:%S'))
            if len(faces) > 6:
                result.append(OutputImgs(faces, target_imgs, timestamps))
        return result

    def get_tracks(self):
        return self.tracks
