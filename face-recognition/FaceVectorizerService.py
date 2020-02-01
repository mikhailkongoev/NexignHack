import cv2
import os
from face_vectorizer import OpenvinoFaceVectorizer
from openvino_detectors.FaceDetector import FaceDetector


class VectorizeError(Exception):
    """
    Ошибка векторизации.
    """


class FaceVectorizerService:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.vectorizer = OpenvinoFaceVectorizer()

    def find_by_save_img(self, img_path):
        target_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        return self.find_by_img(target_image)

    def find_by_img(self, target_image):
        detections = self.face_detector.detect(target_image)
        if len(detections) == 0:
            # cv2.imwrite(result_path, target_image)
            return False, target_image

        all_result = []
        for i, det in enumerate(detections):
            cur_face = self.face_detector.crop(target_image, det)
            # cv2.imwrite("/home/sergej2/PycharmProjects/face/crop_face/"+str(i)+".png", cur_face)
            try:
                cur_result = self.vectorizer.searcher(cur_face, top=1)[0]
            except IndexError:
                raise VectorizeError("Images Not Found")
            folders = cur_result[1].split("/")
            name_img = folders[-1]
            cv2.rectangle(target_image, (det[0], det[1]), (det[2], det[3]), (0, 250, 0), 4)
            cv2.putText(target_image, str(name_img + " ") + str(cur_result[0])[0:4], (det[2], det[3]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), thickness=2)
            all_result.append(cur_result)

        # cv2.imwrite(result_path, target_image)

        return True, target_image

    def find_faces(self, frame):
        detections = self.face_detector.detect(frame)
        return detections

    def get_img_from_det(self, img, detections):
        res = []
        for v in detections:
            left = v[0]
            top = v[1]
            right = v[2]
            bottom = v[3]
            res.append(img[top:bottom, left:right])
        return res


    def get_vector_from_face(self, face):
        return self.vectorizer.face_to_vector(face)


if __name__ == "__main__":
    service = FaceVectorizerService()
    service.load_images("/home/sergej2/For demo/demo_test")

    res1 = service.find_by_save_img(
        "/home/sergej2/For demo/test/10-1.jpg")
    print(res1)
