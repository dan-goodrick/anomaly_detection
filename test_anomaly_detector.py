# USAGE
# python test_anomaly_detector.py --model anomaly_detector.model --image examples/highway_a836030.jpg

# import the necessary packages
from pyimagesearch.features import quantify_image
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained anomaly detection model")
ap.add_argument("-p", "--path", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the anomaly detection model
print("[INFO] loading anomaly detection model...")
model = pickle.loads(open(args["model"], "rb").read())
model.threshold = 'new'
# load the input image, convert it to the HSV color space, and
# quantify the image in the *same manner* as we did during training
for img in os.listdir(args['path']):
	image = cv2.imread(os.path.join(args["path"], img))
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	features = quantify_image(hsv, bins=(3, 3, 3))

	# use the anomaly detector model and extracted features to determine
	# if the example image is an anomaly or not
	preds = model.predict([features])[0]
	label = "anomaly" if preds == -1 else "normal"
	print(f"{img} is {label}")
