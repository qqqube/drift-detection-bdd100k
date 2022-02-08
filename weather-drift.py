import numpy as np
import alibi_detect
import torch
import matplotlib.pyplot as plt
from alibi_detect.cd import ClassifierDrift
from alibi_detect.utils.saving import save_detector
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
import json
import logging
import sys

logging.getLogger().setLevel(logging.DEBUG)

training_weather_count = json.load(open("json_files/train_weather_count.json"))
validation_weather_counts = json.load(open("json_files/val_weather_count.json"))
train_weather_img = json.load(open("json_files/training_weather_names.json"))
val_weather_img = json.load(open("json_files/val_weather_names.json"))

weather_attributes = ["rainy", "snowy", "clear", "overcast", "partly cloudy", "foggy"]
train_sizes = [0.20, 0.40, 0.60, 0.80]

assert sum(training_weather_count.values()) == sum([len(item) for item in train_weather_img.values()])
assert sum(validation_weather_counts.values()) == sum([len(item) for item in val_weather_img.values()])
IMG_SIZE = (720, 1280)


def _eval(classifier, evaluation_set, p_val_lst, detect_lst):
	"""
	Given a trained drift classifier, evaluate
	on evaluation_set and populate lists
	"""
	prediction = classifier.predict(evaluation_set)
	p_val_lst.append(prediction["data"]["p_val"])
	detect_lst.append(prediction["data"]["is_drift"])


def _train(reference_set):
	"""
	Given a reference_set, return a trained ClassifierDrift
	"""
	
	tf.random.set_seed(0)

	model = tf.keras.Sequential(
	    [
	        Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
	        Conv2D(8, 4, strides=2, padding='same', activation=tf.nn.relu),
	        Conv2D(16, 4, strides=2, padding='same', activation=tf.nn.relu),
	        Conv2D(32, 4, strides=2, padding='same', activation=tf.nn.relu),
	        Flatten(),
	        Dense(2, activation='softmax')
	    ]
	  )

	# train with cross validation
	cd = ClassifierDrift(reference_set, model, p_val=.05, epochs=1, n_folds=5)

	return cd



def _get_target_distribution(reference_domain):
	"""
	Return images in validation set as numpy array
	"""
	
	img_lst = val_weather_img[reference_domain]

	lst = []
	for img_file in img_lst:

		img_file_path = os.path.join(os.environ["VAL"], img_file)

		img = np.array(Image.open(img_file_path).resize(IMG_SIZE))

		lst.append(img)

	return np.array(lst)


def _get_reference_distribution(reference_domain, size):
	"""
	Sample `size` images from the training set and
	return as numpy array
	"""

	img_lst = train_weather_img[reference_domain]
	idx_sample = np.random.choice(range(len(img_lst)), size=size, replace=False)
	img_lst = [img_lst[idx] for idx in idx_sample]

	lst = []
	for img_file in img_lst:

		img_file_path = os.path.join(os.environ["TRAIN"], img_file)

		img = np.array(Image.open(img_file_path).resize(IMG_SIZE))

		lst.append(img)

	return np.array(lst)


def _write_json(data, path):
	"""
	Given dictionary (data), write to file specified by path
	"""
	with open(os.path.join("out", path), "w") as f:
		json.dump(data, f)
		f.close()


def parse_args():
	"""
	Parse arguments
	"""

	if len(sys.argv) != 5:
		logging.error("Please specify all required arguments.")
		return

	return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]


if __name__ == "__main__":


	reference_domain, target_domain, ref_out_path, target_out_path = parse_args()
	logging.info("drift detection with reference dist %s and target dist %s", reference_domain, target_domain)

	p_value_ref = []
	p_value_target = []

	detect_ref = []
	detect_target = []

	# get distributions for evaluation
	target_dist_ref = _get_target_distribution(reference_domain)
	target_dist_diff = _get_target_distribution(target_domain)


	for train_size in train_sizes:


		reference_set = _get_reference_distribution(reference_domain, 
			                                        int(train_size * training_weather_count[reference_domain])
			                                        )

		classifier = _train(reference_set)

		# evaluate on unseen images from reference domain
		_eval(classifier, target_dist_ref, p_value_ref, detect_ref)

		# evaluate on unseen images from target domain
		_eval(classifier, target_dist_diff, p_value_target, detect_target)

		# save classifier to models directory
		save_detector(classifier, os.path.join("models", "%s_%s" % (reference_domain, train_size)))

		logging.info("saved trained detector for training_size %s", train_size)

	ref_out_json = {"p_value": p_value_ref, "detect": detect_ref}
	target_out_json = {"p_value": p_value_target, "detect": detect_target}

	_write_json(ref_out_json, ref_out_path)
	_write_json(target_out_json, target_out_path)



	