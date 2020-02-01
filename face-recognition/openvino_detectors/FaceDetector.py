
"""
45 FPS
"""
from openvino_detectors.openvino_detectors import OpenvinoDetector


class FaceDetector(OpenvinoDetector):
    def __init__(self, detection_threshold=0.5):
        super().__init__(cpu_lib="/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so",
                         detector_xml="/home/mikhailkongoev/Desktop/hackaton/NexignHack/face-recognition/openvino_detectors/face-detection/FP32/face-detection-adas-0001.xml",
                         # detector_xml="openvino_detectors/models/person-vehicle-bike-detection-crossroad-0078/person-vehicle-bike-detection-crossroad-0078.xml",
                         detection_threshold=detection_threshold)

    def detect(self, frame):
        height, width = frame.shape[:-1]
        cur, new_height, new_width = self.get_detections(frame)
        detections = cur[self.d_out][0][0]
        result = []
        for _, label, confidence, left, top, right, bottom in detections:
            if label == 1:
                if confidence > self.detection_threshold:
                    left, top, right, bottom = self.convert_detections((left, top, right, bottom), height, width,
                                                                       new_height, new_width)

                    result.append((left, top, right, bottom, float(confidence)))

        return result

    def crop(self, img, detection):
        return img[detection[1]:detection[3], detection[0]:detection[2]]
