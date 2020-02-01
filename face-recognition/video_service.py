import io
import pickle
import sys
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np

from FaceVectorizerService import FaceVectorizerService
import time
import requests

from face_searcher import FaceSearcherService
from online_tracking import PhantomSortTrackerOnline
from visualize import draw_boxes_on_image_online, draw_boxes_on_image_det
import base64


def make_request(url, data):
    """
    Выполнить звапрос
    :param url:
    :param data:
    :return:
    """
    requests.post(
        url,
        data=data
    )


import json
import datetime
from kafka import KafkaProducer

# 172.25.0.1
# 172.17.0.1
# 172.21.120.37
# 10.51.228.24
producer = KafkaProducer(bootstrap_servers=['172.21.120.37:9092'],
                         value_serializer=lambda m: json.dumps(m).encode('utf-8'))
print("")


def send_to_kafka(id_face, similarity, face_photo, video_frame):
    _, face_enc = cv2.imencode('.jpg', face_photo)
    _, video_enc = cv2.imencode('.jpg', video_frame)
    jpg_as_text_face = base64.b64encode(face_enc)
    jpg_as_text_video = base64.b64encode(video_enc)
    producer.send('atomic', {
        'id': str(id_face),
        'basePhoto': str(jpg_as_text_face),
        'cameraPhoto': str(jpg_as_text_video),
        'dateTime': str(datetime.datetime.now()),
        'similarity': str(similarity)})

    producer.flush()


class VideoService:
    def __init__(
            self,
            path_video,
            tracker_config,
            server_address,
            nth_frame=1,
            visualize=False,
            path_to_faces="",
            market_code=None
    ):
        self.path_video = path_video
        self.face_vectorizer = FaceVectorizerService()
        self.tracker = PhantomSortTrackerOnline.from_json(tracker_config)
        self.nth_frame = nth_frame
        self.market_code = market_code
        self.server_address = server_address
        self.visualize = visualize
        if self.visualize:
            self.face_searcher = FaceSearcherService()
            self.face_searcher.load_images(path_to_faces)

    def score(self, detection, center):
        square_coeff = 4
        distance_coeff = 1

        import math

        det_width = detection[2] - detection[0]
        det_height = detection[3] - detection[1]
        (y, x) = center

        square = det_height * det_width
        distance = math.sqrt((detection[0] + det_width / 2 - x) ** 2 + (detection[1] + det_height / 2 - y) ** 2)

        return -(square * square_coeff - distance * distance * distance_coeff)

    def filter_main_face(self, detections, frame):
        if not detections:
            return None

        (height, width) = frame.shape[:-1]
        center = (height / 2, width / 2)

        result_list = {}
        for det in detections:
            result_list[det] = self.score(det, center)

        return sorted(result_list.items(), key=lambda item: item[1])[0][0]


    def run(self):
        cap = cv2.VideoCapture(self.path_video)
        number_frame = 0

        new_video = None
        if self.visualize:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = int(25)

            path_result = "visualize_dir/"
            new_video = cv2.VideoWriter(path_result + "out_hack.avi", fourcc=fourcc, fps=fps,
                                        frameSize=(360, 640))
            # frameSize=(1920, 1080))
        start = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:

            while True:
                ret, frame = cap.read()

                if not ret:
                    target_tracks = self.tracker.get_tracks()

                else:

                    if number_frame % self.nth_frame == 0:
                        detections = self.face_vectorizer.find_faces(frame)
                        # colored_boxes, target_tracks = self.tracker.update_all(detections, number_frame, frame)

                        main_detection = self.filter_main_face(detections, frame)
                        detections = []
                        if main_detection is not None:
                            detections.append(main_detection)

                        faces = self.face_vectorizer.get_img_from_det(frame, detections)
                        for face in faces:
                            face_vector = self.face_vectorizer.get_vector_from_face(face)
                            # cv2.imwrite("/home/sergej/PycharmProjects/FaceHackathon/face_crop/test.png", face)
                            cur = self.face_searcher.searcher(face_vector, top=1, threshold=0.7)
                            if len(cur) > 0:
                                similarity, id_img = cur[0]
                                face = self.face_searcher.face_images[id_img]
                                print(id_img + " " + str(similarity))
                                name_f = id_img.split("/")[-1]
                                send_to_kafka(id_img, similarity, face, frame)
                                # cv2.imwrite("/home/sergej/PycharmProjects/FaceHackathon/result_faces/"+name_f, face)
                            else:
                                send_to_kafka(str(-1), 0, frame, frame)

                        if len(faces) == 0:
                            send_to_kafka(str(0), 0, frame, frame)
                        if self.visualize:
                            vis_frame = draw_boxes_on_image_det(frame, detections)

                if self.visualize:
                    new_video.write(vis_frame)

                number_frame += 1
                if number_frame % 100 == 0:
                    print("Working on " + str(number_frame))
                if not ret:
                    break
                # if number_frame > 1000:
                #     break

        # new_video.release()
        cur = (time.time() - start)
        print(cur)
        print(number_frame / cur)


if __name__ == "__main__":
    # path = "/home/sergej/Downloads/hack2.mp4"
    path = 0
    video_service = VideoService(path, "tracker_config.json", "",
                                 visualize=True, path_to_faces='/home/sergej/faces_all')
    video_service.run()
