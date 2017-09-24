import csv
import cv2
import numpy as np
import sklearn
import math
import matplotlib as mpl
mpl.use('Agg') # use Agg backend in order to be able to save images without a display (AWS EC2 backend)
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Activation, Convolution2D, MaxPooling2D, Dropout, ELU, Reshape
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from random import randint


def replace_image_paths(line, path):
    for camera in range(3):
        source_path = line[camera]
        filename = source_path.split('/')[-1]
        line[camera] = path + '/IMG/' + filename
    return line


def line_is_good_enough(line):
    throttle = float(line[4])

    if throttle < 0.25:
        return False

    return True


def load_csv(paths, file_name):
    lines = []

    for path in paths:
        file = path + file_name

        with open(file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                line = replace_image_paths(line, path)
                if line_is_good_enough(line):
                    lines.append(line)
                    line[3] = round_nearest(float(line[3]), 0.0004)

    return lines


def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))


def extract_feature_and_label_from(augment_dataset, line):
    camera = 0

    if augment_dataset:
        camera = randint(0, 2)

    image = cv2.imread(line[camera])
    label = line[3]

    if augment_dataset:
        correction = 0.2

        if camera == 1 and label > 0:
            label += correction

        if camera == 2 and label < 0:
            label -= correction

    return image, label


def mirror(feature, label):
    mirror_probability = np.random.random()
    if mirror_probability >= 0.5:
        feature = cv2.flip(feature, 1)
        label *= -1.0

    return feature, label


def get_random_translation(maximum_translation):
    return maximum_translation*np.random.uniform() - maximum_translation/2


def translate(feature, label):
    horizontal_translation = get_random_translation(maximum_translation=100)
    vertical_translation = get_random_translation(maximum_translation=40)

    transformation_matrix = np.float32([[1, 0, horizontal_translation], [0, 1, vertical_translation]])
    output_size = (320, 160)
    feature = cv2.warpAffine(feature, transformation_matrix, output_size)
    label += horizontal_translation * 0.004  # we add 0.004 steering angle for every translated pixel
    return feature, label


def modify_brightness(feature):
    feature = cv2.cvtColor(feature, cv2.COLOR_RGB2HSV)
    feature = np.array(feature, dtype=np.float64)

    random_bright = 0.25 + np.random.uniform()

    feature[:, :, 2] = feature[:, :, 2] * random_bright
    feature[:, :, 2][feature[:, :, 2] > 255] = 255

    feature = np.array(feature, dtype=np.uint8)
    return cv2.cvtColor(feature, cv2.COLOR_HSV2RGB)


def provide_augmented_dataset_from(augment_dataset, line):
    feature, label = extract_feature_and_label_from(augment_dataset, line)

    if augment_dataset:
        feature, label = mirror(feature, label)
        feature, label = translate(feature, label)
        feature = modify_brightness(feature)

    return feature, label


def generate_samples(augment_dataset, batch_size, lines):
    features = np.zeros((batch_size, 160, 320, 3))
    labels = np.zeros(batch_size)

    num_samples = len(lines)

    # we will generate as many images as batches
    for batch in range(batch_size):
        line_index = np.random.randint(num_samples)
        line = lines[line_index]

        include_sample = 0
        while include_sample == 0:
            feature, label = provide_augmented_dataset_from(augment_dataset, line)
            if augment_dataset:
                abs_label = abs(label)
                probability = np.random.uniform()
                if abs_label < 0.25 and probability > 0.85\
                        or abs_label >= 0.25 and abs_label < 0.5 and probability > 0.25\
                        or abs_label >= 0.5:
                    features[batch] = feature
                    labels[batch] = label
                    include_sample = 1
            else:
                features[batch] = feature
                labels[batch] = label
                include_sample = 1

    return features, labels


def extract_features_labels_generator(lines, augment_dataset):
    batch_size = 32

    lines = shuffle(lines)

    while 1:  # Loop forever so the generator terminates when all samples are produced
        features, labels = generate_samples(augment_dataset, batch_size, lines)
        yield features, labels


def print_loss(history_loss):
    plt.plot(history_loss.history['loss'])
    plt.plot(history_loss.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('model_loss.png')
    plt.close()


def simple_network_architecture(model):
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


def lenet(model):
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(Dense(84))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


def lenet_with_dropout(model):
    model.add(Convolution2D(6, 5, 5, activation="elu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(84))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


def nvidia(model):
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


def train_with_generator(model, train_lines, validation_lines, train_validation_ratio):
    desired_generated_samples = 20480
    desired_generated_training_samples = desired_generated_samples * train_validation_ratio
    desired_generated_validation_samples = desired_generated_samples * (1 - train_validation_ratio)

    generator_train = extract_features_labels_generator(train_lines, True)
    generator_validation = extract_features_labels_generator(validation_lines, False)
    history = model.fit_generator(generator_train,
                                  samples_per_epoch=desired_generated_training_samples,
                                  validation_data=generator_validation,
                                  nb_val_samples=desired_generated_validation_samples,
                                  verbose=1,
                                  nb_epoch=7,
                                  )
    print_loss(history)
    return model


def add_image_pre_processing_layers(model):
    # values normalisation
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    # cropping in order to remove environment and car
    model.add(Cropping2D(cropping=((62, 23), (0, 0))))
    return model


def print_images(features):
    random_images = shuffle(features)
    number_of_images_to_print = 25
    for i in range(number_of_images_to_print):
        image = np.array(random_images[i], dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        plt.imshow(image)
        plt.savefig("resources/augmented_dataset_examples/image_" + str(i) + ".png")
        plt.close()


def print_images_in_grid(features):
    random_images = shuffle(features)
    number_of_images_to_print = 25
    plt.figure(figsize=(16, 8))
    for i in range(number_of_images_to_print):
        image = np.array(random_images[i], dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        plt.subplot(5, 5, i+1)
        plt.imshow(image)
        plt.axis('off')
    plt.savefig("resources/augmented_dataset_examples/image_grid.png")
    plt.close()


def print_histogram(name, steerings):
    xmin = min(steerings)
    xmax = max(steerings)
    step = 0.01
    num = int(xmax-xmin/step)
    print(xmin)
    print(xmax)
    print(num)

    y, x = np.histogram(steerings, bins=np.linspace(xmin, xmax, num))
    nbins = y.size
    plt.bar(x[:-1], y, width=x[1] - x[0], color='blue', alpha=0.5)
    plt.hist(steerings, bins=nbins, alpha=0.5, normed=True)
    plt.grid(True)
    plt.savefig("resources/" + name + '_histogram.png')
    plt.close()


def get_steerings_from(lines):
    steerings = []
    for line in range(len(lines)):
        steering = float(lines[line][3])
        steerings.append(steering)
    return steerings


def display_dataset_analysis(lines):
    steerings = get_steerings_from(lines)
    print_histogram('original_dataset', steerings)
    features, labels = generate_samples(True, 3000, lines)
    print_histogram('extended_dataset', labels)
    #  print_images(features)
    print_images_in_grid(features)


def get_training_and_validation_samples_from(paths, train_validation_ratio):
    lines = load_csv(paths, '/driving_log.csv')
    lines = shuffle(lines)

    display_dataset_analysis(lines)

    return train_test_split(lines, test_size=(1 - train_validation_ratio), random_state=0)


paths = [
    'resources/dataset/track_1_pre_recorded'
    # 'resources/dataset/track_1_multiple_laps',
    # 'resources/dataset/track_1_multiple_laps_counter_clockwise'
    # 'resources/dataset/track_2_counter_clockwise',
    # 'resources/dataset/track_2_multiple_laps'
]

model = Sequential()
model = add_image_pre_processing_layers(model)

# model = simple_network_architecture(model)
# model = lenet(model)
# model = lenet_with_dropout(model)
model = nvidia(model)

train_validation_ratio = 0.8
train_lines, validation_lines = get_training_and_validation_samples_from(paths, train_validation_ratio)
model = train_with_generator(model, train_lines, validation_lines, train_validation_ratio)

model.save('model.h5')

exit()
