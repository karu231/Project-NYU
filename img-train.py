# load and prepare stamp dataset and save to file
import os
from skimage.io import imread
import skimage.transform
from os import listdir
from numpy import zeros
from numpy import asarray
from numpy import savez_compressed
from pandas import read_csv
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import pandas as pd

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
# dict that maps labels to tags, and the reverse
labels_map = {labels[i]: i for i in range(len(labels))}
inv_labels_map = {i: labels[i] for i in range(len(labels))}


# create a one hot encoding for one list of tags
def one_hot_encode(tags, mapping):
	# create empty vector
	encoding = zeros(len(mapping), dtype='uint8')
	# mark 1 for each tag in the vector
	for tag in tags:
		encoding[mapping[tag]] = 1
	return encoding


# load all images into memory
def load_dataset(path, file_mapping, tag_mapping):
	photos, targets = list(), list()
	# enumerate files in the directory
	print(folder)
	for filename in listdir(folder):
		# load image
		photo = load_img(path + filename, target_size=(128, 128))
		# convert to numpy array
		photo = img_to_array(photo, dtype='uint8')
		# get tags
		print('filename',filename)
		print(filename[:-4])
		#print(file_mapping[1])
		#tags = file_mapping[int(filename[:-4])]
		tags = file_mapping[(filename[:-4] + '.jpg')]
		# one hot encode tags
		target = one_hot_encode(tags, tag_mapping)
		# store
		photos.append(photo)
		targets.append(target)
	X = asarray(photos, dtype='uint8')
	y = asarray(targets, dtype='uint8')
	return X, y

# load the mapping file
filename = 'map.csv'
mapping_csv = read_csv(filename)
# create a mapping of tags to integers
tag_mapping = labels_map
# create a mapping of filenames to tag lists
file_mapping = mapping
print('tag_mapping',tag_mapping)
print('file_mapping',file_mapping)
# load the jpeg images
folder =  'Dataset/stamp/consol/'
X, y = load_dataset(folder, file_mapping, tag_mapping)
print(X.shape, y.shape)
# save both arrays to one file in compressed format
savez_compressed('stamp_data_gen.npz', X, y)

# load prepared stamp dataset
from numpy import load
data = load('stamp_data_gen.npz')
X, y = data['arr_0'], data['arr_1']
print('Loaded: ', X.shape, y.shape)

# baseline model for the stamp dataset
import sys
from numpy import load
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras import backend
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD


# load train and test dataset
def load_dataset():
	# load dataset
	data = load('stamp_data_gen.npz')
	X, y = data['arr_0'], data['arr_1']
	# separate into train and test datasets
	trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.3, random_state=1)
	print(trainX.shape, trainY.shape, testX.shape, testY.shape)
	return trainX, trainY, testX, testY


# calculate fbeta score for multi-class/label classification
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


# define cnn model
def define_model(in_shape=(128, 128, 3), out_shape=4):
	model = Sequential()
	model.add(
		Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=in_shape))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(out_shape, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[fbeta])
	return model


# plot diagnostic learning curves
def plot_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Fbeta')
	pyplot.plot(history.history['fbeta'], color='blue', label='train')
	pyplot.plot(history.history['val_fbeta'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()


# run the train model img for evaluating a model
def train_model_img():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# create data generator
	#datagen = ImageDataGenerator(rescale=1.0 / 255.0)
	train_datagen = ImageDataGenerator(rescale=1.0/255.0, shear_range=0.2,zoom_range=0.2,horizontal_flip=True, vertical_flip=True, rotation_range=90)
	test_datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
	train_it = train_datagen.flow(trainX, trainY, batch_size=128)
	test_it = test_datagen.flow(testX, testY, batch_size=128)
	# define model
	model = define_model()
	# fit model
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
								  validation_data=test_it, validation_steps=len(test_it), epochs=100, verbose=1)
	# evaluate model
	loss, fbeta = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> loss=%.3f, fbeta=%.3f' % (loss, fbeta))
	# learning curves
	plot_diagnostics(history)
	# save model
	model.save('final_model.h5')

# entry point, run the train img
train_model_img()