from __future__ import print_function

import os
from queue import PriorityQueue

import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IEPlugin
from scipy import spatial

from face_aligner import FaceAligner


class OpenvinoFaceVectorizer:
    def __init__(self,
                 cpu_lib="/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so",
                 landmarks_xml="openvino_detectors/landmarks-regression/FP32/model.xml",
                 features_xml="openvino_detectors/face-reidentification/FP32/model.xml"):

        # Plugin initialization for specified device and load extensions library if specified
        plugin = IEPlugin(device="CPU")
        plugin.add_cpu_extension(cpu_lib)

        # Read landmarks IR
        landmarks_bin = os.path.splitext(landmarks_xml)[0] + ".bin"
        log.info("Loading landmarks network files:\n\t{}\n\t{}".format(landmarks_xml, landmarks_bin))
        landmarks_net = IENetwork.from_ir(model=landmarks_xml, weights=landmarks_bin)

        # Read features IR
        features_bin = os.path.splitext(features_xml)[0] + ".bin"
        log.info("Loading features network files:\n\t{}\n\t{}".format(features_xml, features_bin))
        features_net = IENetwork.from_ir(model=features_xml, weights=features_bin)
        self.l_in = next(iter(landmarks_net.inputs))
        self.l_out = next(iter(landmarks_net.outputs))
        landmarks_net.batch_size = 1

        self.f_in = next(iter(features_net.inputs))
        self.f_out = next(iter(features_net.outputs))
        features_net.batch_size = 1
        cur = landmarks_net.inputs[self.l_in]
        self.l_n = cur.layout
        self.l_c, self.l_h, self.l_w = cur.shape[1:]
        # self.l_n = NCHW it is 1
        self.l_images = np.ndarray(shape=(1, self.l_c, self.l_h, self.l_w))

        cur = features_net.inputs[self.f_in]
        self.f_n = cur.layout
        self.f_c, self.f_h, self.f_w = cur.shape[1:]

        self.f_images = np.ndarray(shape=(1, self.f_c, self.f_h, self.f_w))

        # Loading models to the plugin
        log.info("Loading models to the plugin")
        self.l_exec_net = plugin.load(network=landmarks_net)
        self.f_exec_net = plugin.load(network=features_net)

        self.face_aligner = FaceAligner(face_width=self.f_w, face_height=self.f_h)
        self.vectors = {}

    def face_to_vector(self, face):
        height, width = face.shape[:-1]
        landmark_face = cv2.resize(face, (self.l_w, self.l_h))
        self.l_images[0] = landmark_face.transpose((2, 0, 1))
        l_res = np.squeeze(self.l_exec_net.infer(inputs={self.l_in: self.l_images})[self.l_out])
        for i in range(10):
            if i % 2 == 0:
                l_res[i] = width * l_res[i]
            else:
                l_res[i] = height * l_res[i]
        aligned_face = self.face_aligner.align(face, l_res)
        self.f_images[0] = aligned_face.transpose((2, 0, 1))
        # self.f_images[0] = cv2.resize(face, (self.f_w, self.f_h)).transpose((2, 0, 1))
        f_res = np.squeeze(self.f_exec_net.infer(inputs={self.f_in: self.f_images})[self.f_out])
        # print(f_res)
        # cv2.imshow('frame', face)
        # cv2.waitKey(1000)
        return np.array(f_res)

    def searcher(self, face_img, top=3):

        face_vector = self.face_to_vector(face_img)
        nearest = PriorityQueue()

        for id_people, faces in self.vectors.items():
            for face in faces:
                similarity = self.face_similarity(face, face_vector)
                nearest.put((similarity, id_people))
                if nearest.qsize() > top:
                    nearest.get()
                # if similarity > max_similarity:
                #     max_similarity = similarity
                #     max_id = id_people
        res = sorted(nearest.queue, key = lambda x:x[0], reverse=True)
        return res

    def add_face(self, face, face_name):
        self.vectors[face_name] = [self.face_to_vector(face)]

    def face_similarity(self, v1, v2):
        return 1.0 - spatial.distance.cosine(v1, v2)
