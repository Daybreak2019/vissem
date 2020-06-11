
import cv2
import numpy as np
import os
import logging
import pickle


from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential

logger = logging.getLogger(__name__)

def get_files(data_path, max_classes=0, min_samples_per_class=0):
    """Returns a list of image files and classes.
  Args:
      :param data_path: A directory containing a set of subdirectories representing class names. Each subdirectory should contain PNG or JPG encoded images.
      :param min_samples_per_class:
    :param max_classes:
  Returns:
    A list of image file paths, relative to `data_path` and the list of
    subdirectories, representing class names.
  """
    folders = [name for name in os.listdir(data_path) if
               os.path.isdir(os.path.join(data_path, name))]

    if len(folders) == 0:
        raise ValueError(data_path + " does not contain valid sub directories.")
    directories = []
    for folder in folders:
        directories.append(os.path.join(data_path, folder))

    folders = sorted(folders)
    id2label = {}

    i = 0
    c = 0
    total_files = []
    for folder in folders:
        dir = os.path.join(data_path, folder)
        files = os.listdir(dir)
        if min_samples_per_class > 0 and len(files) < min_samples_per_class:
            continue

        for file in files:
            path = os.path.join(dir, file)
            total_files.append([path, i])
        id2label[i] = folder
        i += 1

        if 0 < max_classes <= c:
            break
        c += 1

    return np.array(total_files), id2label


class ElementClassifier(object):

    def __init__(self, model_path=None, load=False, classes=None, id2lebel=None):
        self.model_path = model_path
        self.image_size = (28, 28)
        self.image_channel = 1
        self.id2label = id2lebel
        self.classes = classes
        self.load_meta()
        self.model = self.build_lenet_model(self.classes)
        if load and self.model_path is not None and os.path.exists(model_path):

            try:
                self.load()
            except Exception as e:
                logger.exception(e)

    def preprocess(self, img):
        img = cv2.resize(img, self.image_size)
        if self.image_channel == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.reshape((self.image_size[0], self.image_size[1], 1))
        img = img / 255.0
        return img

    def build_model(self, classes):
        assert classes is not None, "num_classes can not be None"
        print("Num classes: {}".format(classes))
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
                         input_shape=(self.image_size[0], self.image_size[1], self.image_channel)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_lenet_model(self, classes):
        # initialize the model
        model = Sequential()

        # first set of CONV => RELU => POOL
        model.add(Conv2D(20, kernel_size=(5, 5), padding='same', activation='relu',
                         input_shape=(self.image_size[0], self.image_size[1], self.image_channel)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL
        model.add(Conv2D(50, kernel_size=(5, 5), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500, activation="relu"))

        # softmax classifier
        model.add(Dense(classes, activation="softmax"))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, train_x, train_y, test_x=None, test_y=None, epochs=200, validation_split=0.1, batch_size=32):
        validation_data = None if test_x is None else (test_x, test_y)
        history = self.model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1,
                                 validation_data=validation_data, validation_split=validation_split)
        # print("Loss history", history.history)

        if test_x is not None:
            self.model.evaluate(test_x, test_y)

    def save(self):
        self.model.save_weights(self.model_path, overwrite=True)
        meta_file = os.path.join(os.path.dirname(self.model_path), "meta.pkl")
        with open(meta_file, 'wb') as f:
            pickle.dump({"id2label": self.id2label, "classes": self.classes}, f)

    def load_meta(self):
        meta_file = os.path.join(os.path.dirname(self.model_path), "meta.pkl")
        if os.path.exists(meta_file):
            with open(meta_file, 'rb') as f:
                data = pickle.load(f)
                self.id2label = data["id2label"]
                self.classes = data["classes"]

    def load(self):
        self.model.load_weights(self.model_path)

    def predict(self, img, preprocess=True):
        if preprocess:
            img = self.preprocess(img)
        if img.ndim == 3:
            img = img.reshape((-1, self.image_size[0], self.image_size[1], self.image_channel))
        return self.model.predict(img)

def classifier_train():
    data_dir = "data/training_data/"
    model_path = os.path.join("data/model/", "classifier.model")
    images, id2label = get_files(data_dir)
    print(id2label)

    model = ElementClassifier(model_path=model_path, load=False, classes=np.max(list(id2label.keys())) + 1,
                              id2lebel=id2label)

    X = []
    y = []
    for path, label in images:
        try:
            img = cv2.imread(path)
            img = model.preprocess(img)
            X.append(img)
            y.append(label)
        except:
            pass
    X = np.array(X)
    y = np.array(y).astype(np.int32)
    n_values = np.max(y) + 1
    y = np.eye(n_values)[y]

    model.fit(X, y, epochs=50)
    model.save()

