
import dlib
import cv2
from imutils import face_utils

image = cv2.imread("images/VOTREPORTRAIT.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

dots_size = 5

rects = detector(gray, 1)
for (i, rect) in enumerate(rects):
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 
	for (x, y) in shape:
		cv2.circle(image, (x, y), dots_size, (0, 255, 0), -1)
 
cv2.imshow("Sortie :", image)
cv2.waitKey(0)
