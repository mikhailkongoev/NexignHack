from __future__ import print_function

import os

import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin


class OpenvinoDetector:
    def __init__(self, cpu_lib, detector_xml, detection_threshold):
        """
        Initialize openvino detector, load configuration network and weights
        :param cpu_lib:
        :param detector_xml:
        :param detection_threshold:
        """
        # Plugin initialization for specified device and load extensions library if specified
        plugin = IEPlugin(device="CPU")
        plugin.add_cpu_extension(cpu_lib)

        # Read detector IR
        detector_bin = os.path.splitext(detector_xml)[0] + ".bin"
        detector_net = IENetwork.from_ir(model=detector_xml, weights=detector_bin)

        self.d_in = next(iter(detector_net.inputs))
        self.d_out = next(iter(detector_net.outputs))
        detector_net.batch_size = 1

        # Read and pre-process input images
        self.d_n, self.d_c, self.d_h, self.d_w = detector_net.inputs[self.d_in].shape
        self.d_images = np.ndarray(shape=(self.d_n, self.d_c, self.d_h, self.d_w))

        # Loading models to the plugin
        self.d_exec_net = plugin.load(network=detector_net)

        self.detection_threshold = detection_threshold

    def get_detections(self, frame):
        """
        Run network on frame and get detections in raw format

        :param frame:
        :return: output network in raw format
        Use method detect to receive bounding boxes (left, top, right, bottom, confidence)
        """
        height, width = frame.shape[:-1]
        if height * self.d_w > self.d_h * width:
            new_width = self.d_w * height / self.d_h
            new_height = height
            border_size = int((new_width - width) / 2)
            frame = cv2.copyMakeBorder(frame, top=0, bottom=0, left=border_size, right=border_size,
                                       borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif height * self.d_w < self.d_h * width:
            new_width = width
            new_height = self.d_h * width / self.d_w
            border_size = int((new_height - height) / 2)
            frame = cv2.copyMakeBorder(frame, top=border_size, bottom=border_size, left=0, right=0,
                                       borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            new_width = width
            new_height = height

        if (new_width, new_height) != (self.d_w, self.d_h):
            d_frame = cv2.resize(frame, (self.d_w, self.d_h))
        else:
            d_frame = frame

        # Change data layout from HWC to CHW
        self.d_images[0] = d_frame.transpose((2, 0, 1))

        d_res = self.d_exec_net.infer(inputs={self.d_in: self.d_images})
        return d_res, new_height, new_width

    def convert_detections(self, det, height, width, new_height, new_width):
        left, top, right, bottom = det
        left = max(0, int(left * new_width - (new_width - width) / 2))
        right = min(int(right * new_width - (new_width - width) / 2), width - 1)

        top = max(0, int(top * new_height - (new_height - height) / 2))
        bottom = min(int(bottom * new_height - (new_height - height) / 2), height - 1)


        return left, top, right, bottom









