# import argparse
from curses import raw
import imutils
import time
import dlib
import cv2
from os import listdir
from help import convert_and_trim_bb
from skimage import io
import numpy

# from skimage.io import imread_collection
# construct the argument parser and parse the arguments

def detect_heads(path_img, dlib_path, upsample_val):
		
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-i", "--image", type=str, required=True,
	# 	default="/data/",
	# 	help="path to input image")
	# ap.add_argument("-m", "--model", type=str,
	# 	default="mmod_human_face_detector.dat",
	# 	help="path to dlib's CNN face detector model")
	# ap.add_argument("-u", "--upsample", type=int, default=1,
	# 	help="# of times to upsample")
	# args = vars(ap.parse_args())

	# load dlib's CNN face detector
	print("[INFO] loading CNN face detector...")
	detector = dlib.cnn_face_detection_model_v1(dlib_path)
	# load the input image from disk, resize it, and convert it from
	# BGR to RGB channel ordering (which is what dlib expects)
	# files = listdir(args["image"])
	files = listdir(path_img)
	# print(files) 

	raw_imgs = []
	img_path = "./data/"
	for file in files:
		# make sure file is an image
		if file.endswith(('jpg','jpeg')):
			raw_imgs.append((img_path+file))


	currentframe = 0
	for img in raw_imgs:
		image = cv2.imread(img)
		image = imutils.resize(image, width=600)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# perform face detection using dlib's face detector
		start = time.time()
		print("[INFO[ performing face detection with dlib...")
		results = detector(rgb, upsample_val)
		end = time.time()
		print("[INFO] face detection took {:.4f} seconds".format(end - start))


		# convert the resulting dlib rectangle objects to bounding boxes,
		# then ensure the bounding boxes are all within the bounds of the
		# input image
		boxes = [convert_and_trim_bb(image, r.rect) for r in results]
		print(currentframe)
		#loop over the bounding boxes
		for (x, y, w, h) in boxes:
			# draw the bounding box on our image
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# show the output image
		# cv2.imshow("Output", image)
		name = './img_head_shots/' + str(currentframe) + '.jpg'
		# outpath = "./output/Hanif_Save.jpg"
		cv2.imwrite(name, image)
		currentframe += 1
		# image = ""
		image = numpy.empty([2,2])
		# print(currentframe)
		cv2.waitKey(5)

