# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

from FaceDetector import BlazeFaceDetector

class AntispoofingTool:
	'''Основной класс работы приложения'''

	def start_stream(self, path=-1, frame_w=800, frame_h=600):
	    ''' init and setup webcam '''
	    stream = cv2.VideoCapture(path)
	    if stream.isOpened():
	        stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
	        stream.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
	        stream.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
	        stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

	    return stream
	def __init__(self):
		# Детектор лиц
		self.face_detector = BlazeFaceDetector('./resources/face_detector/face_detection_front.tflite',
									'./resources/face_detector/anchors.npy' )

		# Захват видео-потока (здесь камера 0)
		frame_w = 800
		frame_h = 600
		self.cap = self.start_stream(0, frame_w, frame_h)

		device_id = 0   #  Номер видео-карты
		self.anti_spoof_model_dir = "./resources/anti_spoof_models"  #  папка с весами для анти спуфинга(Их две. 
		# В оригинале использовались две и результат их среднее значение. Но это долго. Поэтому тут одна из них.)

		self.model_test = AntiSpoofPredict(device_id)   #  Объект модели антиспуфинга
		self.image_cropper = CropImage()  #  Используется для предикта приводя картинку к нужному разрешению.

	def run(self, image):
		'''Основная функция обработки каждого кадра'''

		image_rgb = image[..., ::-1].copy()
		try:
			#image_bbox = face_detector.detect_faces(image)[0]['box']
			face_detector_result = self.face_detector.predict_on_image(image_rgb)
			image_bbox = face_detector_result[0]['bbox']

			prediction = np.zeros((1, 3))
			test_speed = 0


			# sum the prediction from single model's result
			#for model_name in os.listdir(model_dir):
			h_input, w_input, model_type, scale = parse_model_name("4_0_0_80x80_MiniFASNetV1SE.pth")
			param = {
				"org_img": image_rgb,
				"bbox": image_bbox,
				"scale": scale,
				"out_w": w_input,
				"out_h": h_input,
				"crop": True,
			}
			if scale is None:
				param["crop"] = False
			img = self.image_cropper.crop(**param)


			print(img.shape)

			cv2.imshow("cropped_face", img)

			start = time.time()
			prediction += self.model_test.predict(img, os.path.join(self.anti_spoof_model_dir, "4_0_0_80x80_MiniFASNetV1SE.pth"))
			print(f"1 predictions | : {prediction}")
			test_speed += time.time()-start

			print(f"all predictions = {prediction}")
			# draw result of prediction
			label = np.argmax(prediction)
			value = prediction[0][label]/2
			result_text = ""
			if label == 1:
				print("Image 'rf' is Real Face. Score: {:.2f}.".format(value))
				result_text = "Face"
				color = (0, 255, 0)
			else:
				print("Image 'rf' is Fake Face. Score: {:.2f}.".format(value))
				result_text = "Foto"
				color = (0, 0, 255)
			print("Prediction cost {:.2f} s".format(test_speed))
			cv2.rectangle(
				image,
				(image_bbox[0], image_bbox[1]),
				(image_bbox[2], image_bbox[3]),
				color, 2)
			cv2.putText(
				image,
				result_text,
				(image_bbox[0], image_bbox[1] - 5),
				cv2.FONT_HERSHEY_COMPLEX, 1*image.shape[0]/1024, color)
		except Exception as e:
			print("Ooops")
			print(e)


if __name__ == "__main__":

	ast = AntispoofingTool()

	while True:
		_, img = ast.cap.read()
		if _:		
			
			ast.run(img)

			cv2.imshow("presentation", cv2.resize(img, (640,480)))
		
		else:
			print("It not capture")
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	ast.cap.release()
	cv2.destroyAllWindows()


