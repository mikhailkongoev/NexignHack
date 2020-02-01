import base64
import json
import os
import traceback
from pprint import pprint
from queue import PriorityQueue

import numpy as np
import cv2
from scipy import spatial

from face_vectorizer import OpenvinoFaceVectorizer
from openvino_detectors.FaceDetector import FaceDetector
from collections import Counter


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class FaceSearcherService:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.vectorizer = OpenvinoFaceVectorizer()
        self.vectors = {}
        self.count_persons = 0
        self.face_images = {}

    def load_images(self, path_to_images):
        ids = os.listdir(path_to_images)
        for id in ids:
            sub_img_dir = os.path.join(path_to_images, id)
            if not os.path.isdir(sub_img_dir):
                continue
            face_names = []
            for face_name in os.listdir(sub_img_dir):
                face_names.append(os.path.join(path_to_images, id, face_name))

            for face_name in face_names:
                self.add_new_image(face_name, face_name)

        print("Database created len =", len(self.vectors))

    def add_new_image(self, img_path, target_img_path):
        face = cv2.imread(img_path, cv2.IMREAD_COLOR)

        detections = self.face_detector.detect(face)
        if len(detections) == 0:
            return False

        for i, det in enumerate(detections):
            cur_face = self.face_detector.crop(face, det)
            self.vectors[target_img_path] = [self.vectorizer.face_to_vector(cur_face)]
            self.face_images[target_img_path] = face
        return True

    def face_similarity(self, v1, v2):
        return 1.0 - spatial.distance.cosine(v1, v2)

    def np_array_to_base64(self, image):
        retval, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer)
        return jpg_as_text

    def searcher_for_demo(self, output_imgs):
        for output_img in output_imgs:
            cur, index_img_from_camera = self.batch_searcher(output_img.vectors)
            if cur != -1:
                face_from_database, _ = cur[0]
                similarity = index_img_from_camera[0]
                index_img_from_camera = index_img_from_camera[1]
                self.count_persons += 1
                cv2.imwrite("media/folder_results/visualize/"+str(self.count_persons)+".png", output_img.imgs[index_img_from_camera])
                cv2.imwrite("media/folder_results/visualize/"+str(self.count_persons)+"_face.png", self.face_images[face_from_database])
                print(str(self.count_persons) + " "+str(similarity) + " " + face_from_database)


    def batch_searcher(self, face_vectors, top=3):
        counter = Counter()
        max_number = {}
        for i, face_vector in enumerate(face_vectors):
            result = self.searcher(face_vector)
            for v in result:
                face_name_path_for_counter = [v[1]]
                face_name_path = v[1]
                counter.update(face_name_path_for_counter)
                if face_name_path in max_number:
                    cur_similarity, index = max_number[face_name_path]
                    if v[0] > cur_similarity:
                        max_number[face_name_path] = (v[0], i)
                else:
                    max_number[face_name_path] = (v[0], i)

        if len(counter) == 0:
            return -1, -1
        cur = counter.most_common(1)
        return cur, max_number[cur[0][0]]

    def searcher(self, face_vector, top=3, threshold=0.1):

        nearest = PriorityQueue()

        for id_people, faces in self.vectors.items():
            for face in faces:
                similarity = self.face_similarity(face, face_vector)
                if similarity > threshold:
                    nearest.put((similarity, id_people))
                    if nearest.qsize() > top:
                        nearest.get()
                # if similarity > max_similarity:
                #     max_similarity = similarity
                #     max_id = id_people
        res = sorted(nearest.queue, key=lambda x: x[0], reverse=True)
        return res
