# make a prediction for a new image
#load the necessary packages
import os
from skimage.io import imread
from pandas import read_csv
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import CustomObjectScope

# convert a prediction to Lables
def predict_to_map(inv_map, prediction):
    # round probabilities to {0, 1}
    values = prediction.round()
    print('values',values)
    # collect all predicted tags
    tags = [inv_map[i] for i in range(len(values)) if values[i] == 1.0]
    print('tags',tags)
    return tags


# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(128, 128))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 128, 128, 3)
    # center pixel data
    img = img.astype('float32')
    #img = img - [123.68, 116.779, 103.939]
    return img


# baseline model for the stamp dataset
from keras import backend
#define Fbeta for multibale classification score.
def fbeta(y_true, y_pred, beta=2):
	# clip predictions
	y_pred = backend.clip(y_pred, 0, 1)
	# calculate elements
	tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
	fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
	fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
	# calculate precision
	p = tp / (tp + fp + backend.epsilon())
	# calculate recall
	r = tp / (tp + fn + backend.epsilon())
	# calculate fbeta, averaged across each class
	bb = beta ** 2
	fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
	return fbeta_score

# load an image and predict the class
def predict_img(inv_map):
    # load the image
    img = load_image('001-001-0001-002-1.tiff')
    # load model
    custom_objects = {'fbeta': fbeta}
    model = load_model('final_model.h5', custom_objects = custom_objects)
    #with CustomObjectScope({'fbeta': fbeta}):
    #    model = load_model('final_model.h5')
    # predict the class
    result = model.predict(img)
    print('result',result)
    print(result[0])
    # map prediction to tags
    tags = predict_to_map(inv_map, result[0])
    print(tags)


# load the mapping file
def load_data(data_dir):
    """Loads a data set and returns two lists:

    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    print(data_dir)
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    print(directories)
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    filnm = []
    mapping = dict()
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".jpg")]
        file = [os.path.join( f)
					  for f in os.listdir(label_dir) if f.endswith(".jpg")]
        print('file',file_names)
        print(file)
        for i in range(len(file)):
            name, tags = file[i], d
            mapping[name] = tags.split(' ')
        print('mapping',mapping)
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(imread(f))
            #labels.append(int(d))
            labels.append(d)
        for fil in file:
            filnm.append(fil)
    return images, labels,filnm,mapping

# Load training and testing datasets.
ROOT_PATH = "Dataset"
train_data_dir = os.path.join(ROOT_PATH,"stamp/generate")
images, labelb,filnm,mapping = load_data(train_data_dir)
print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labelb)), len(images)))

# create a set of all known tags
labels = set()
# convert set of labels to a list to list
labels = list(set(labelb))
# order set alphabetically
labels.sort()
# dict that maps labels to integers, and the reverse
labels_map = {labels[i]: i for i in range(len(labels))}
inv_labels_map = {i: labels[i] for i in range(len(labels))}
# entry point, run the example
predict_img(inv_labels_map)