import numpy as np
from lesson_functions import *
from random import shuffle
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split


def extract_features_from_dataset(fep: FeatureExtractionParameters, sample_size=0):
    cars = glob.glob('dataset/vehicles/**/*.png', recursive=True)
    notcars = glob.glob('dataset/non-vehicles/**/*.png', recursive=True)

    if sample_size > 0:
        shuffle(cars)
        shuffle(notcars)

        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]

    car_features = extract_features(fep, cars)
    notcar_features = extract_features(fep, notcars)
    return car_features, notcar_features


def extract_features_labels(car_features, notcar_features):
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    return X_scaler, scaled_X, y


def extract_features_from(X_scaler, car_features):
    # Apply the scaler to X
    return X_scaler.transform(car_features)


def split_data(features_x, labels_y):
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    return train_test_split(features_x, labels_y, test_size=0.2, random_state=rand_state)


def get_prediction_accuracy(correct_values, predicted_values):
    number_correct_matches = np.sum(correct_values == predicted_values)
    accuracy = (number_correct_matches / len(correct_values)) * 100
    return "{0:.2f}".format(accuracy)


def extract_features_for_single_image(fep: FeatureExtractionParameters, X_scaler, image):
    feature = extract_features_from_image(fep, image)
    black_car_features = [feature]
    return extract_features_from(X_scaler, black_car_features)


def sliding_windows_dynamic_size(img, xy_window=(410, 410), y_start=230, scale=0.9, vertical_slices=4):
    window_list = []

    image_width = img.shape[1]
    image_height = img.shape[0]

    number_of_windows = int(image_width / xy_window[0])
    pixels_left = int((image_width - xy_window[0]*number_of_windows) / 2)

    overlap = 0.5

    for vs in range(vertical_slices):
        for hd in range(number_of_windows):
            startx = pixels_left + hd * xy_window[0]
            starty = y_start
            endx = pixels_left + ((hd + 1) * xy_window[0])
            endy = y_start + xy_window[1]

            overlap_size = int((endx - startx) * overlap)

            window_list.append(((startx, starty), (endx, endy)))

            if endx+overlap_size <= img.shape[1]:
                window_list.append(((startx+overlap_size, starty), (endx+overlap_size, endy)))

            if endy+overlap_size <= img.shape[0]:
                window_list.append(((startx, starty+overlap_size), (endx, endy+overlap_size)))

        y_start += int(((xy_window[0] - (xy_window[0] * scale)) / 2))
        xy_window = tuple(map(lambda x: int(x * scale), xy_window))
        number_of_windows = int(image_width / xy_window[0])
        pixels_left = int((image_width - xy_window[0] * number_of_windows) / 2)

    return window_list


def calculate_boxes(image, y_start_stop=[None, None], xy_window_minimum=(64, 64), xy_window_maximum=(400, 40),
                    number_of_steps=10, overlap=(0.5, 0.5)):
    boxes = []
    increment = int((xy_window_maximum[0] - xy_window_minimum[0]) / number_of_steps)

    for step in range(number_of_steps+1):
        xy_window = tuple(map(lambda x: x + step*increment, xy_window_minimum))
        slide_window(image, boxes, y_start_stop=y_start_stop, xy_window=xy_window, xy_overlap=overlap)

    return boxes


def find_cars_in_boxes(img, svc, X_scaler, boxes, fep: FeatureExtractionParameters):
    box_with_car = []

    for box in boxes:
        ytop = box[0][1]
        ybottom = box[1][1]
        xleft = box[0][0]
        xright = box[1][0]

        image_patch = img[ytop:ybottom, xleft:xright]
        subimg = cv2.resize(
            image_patch,
            (fep.shape[0], fep.shape[1])
        )

        features = extract_features_for_single_image(fep, X_scaler, subimg)
        prediction = svc.predict(features)
        if prediction == 1:
            box_with_car.append(box)

    return box_with_car


def find_cars(img, svc, X_scaler, fep: FeatureExtractionParameters, scale=1.0, y_start_stop=[None, None], window_size=64):
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]

    img_tosearch = img[y_start_stop[0]:y_start_stop[1], :, :]
    ctrans_tosearch = convert_color(img_tosearch, color_space=fep.color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    # Compute individual channel HOG features for the entire image
    hog = []
    if fep.hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        hog.append(get_hog_features(ch1, fep.orient, fep.pix_per_cell, fep.cell_per_block, feature_vec=False))
        hog.append(get_hog_features(ch2, fep.orient, fep.pix_per_cell, fep.cell_per_block, feature_vec=False))
        hog.append(get_hog_features(ch3, fep.orient, fep.pix_per_cell, fep.cell_per_block, feature_vec=False))
    else:
        ch = ctrans_tosearch[:, :, fep.hog_channel]
        hog.append(get_hog_features(ch, fep.orient, fep.pix_per_cell, fep.cell_per_block, feature_vec=False))

    # Define blocks and steps as above
    nxblocks = (ctrans_tosearch.shape[1] // fep.pix_per_cell) - fep.cell_per_block + 1
    nyblocks = (ctrans_tosearch.shape[0] // fep.pix_per_cell) - fep.cell_per_block + 1

    nblocks_per_window = (window_size // fep.pix_per_cell) - fep.cell_per_block + 1

    nxsteps = (nxblocks - nblocks_per_window) // fep.cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // fep.cells_per_step

    box_car_detected = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * fep.cells_per_step
            xpos = xb * fep.cells_per_step

            xleft = xpos * fep.pix_per_cell
            ytop = ypos * fep.pix_per_cell

            # Extract the image patch
            image_patch = ctrans_tosearch[ytop:ytop + window_size, xleft:xleft + window_size]
            subimg = cv2.resize(
                image_patch,
                (fep.shape[0], fep.shape[1])
            )

            file_features = []
            if fep.spatial_feat is True:
                spatial_features = bin_spatial(subimg, size=fep.spatial_size)
                file_features.append(spatial_features)
            if fep.hist_feat is True:
                # Apply color_hist()
                hist_features = color_hist(subimg, nbins=fep.hist_bins)
                file_features.append(hist_features)
            if fep.hog_feat is True:
                # Extract HOG for this patch
                if fep.hog_channel == 'ALL':
                    hog_feat1 = hog[0][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat2 = hog[1][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat3 = hog[2][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                else:
                    hog_features = hog[0][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

                file_features.append(hog_features)

            # Scale features and make a prediction
            features = np.hstack(file_features).ravel().reshape(1, -1)
            # features = [np.concatenate(file_features)]
            test_features = X_scaler.transform(features)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window_size * scale)

                box_car_detected.append(
                    (
                        (xbox_left, ytop_draw + y_start_stop[0]),
                        (xbox_left + win_draw, ytop_draw + win_draw + y_start_stop[0])
                    )
                )

    return box_car_detected
