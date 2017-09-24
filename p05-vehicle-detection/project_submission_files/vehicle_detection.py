from helper_functions import *
from scipy.ndimage.measurements import label
from sklearn.svm import LinearSVC
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# Extract HOG and color features from the dataset

feature_extraction_parameters = FeatureExtractionParameters(color_space='YCrCb', orient=8, pix_per_cell=8,
                                                            cell_per_block=2, hog_channel='ALL', spatial_size=(16, 16),
                                                            hist_bins=32)


def delete_stored_classifier():
    if os.path.exists("svc_pickle.p"):
        os.remove("svc_pickle.p")


def load_stored_classifier():
    dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["x_scaler"]
    return svc, X_scaler


def train():
    delete_stored_classifier()
    car_features, notcar_features = extract_features_from_dataset(feature_extraction_parameters, sample_size=0)

    # Train a classifier

    X_scaler, features_x, labels_y = extract_features_labels(car_features, notcar_features)
    X_train, X_test, y_train, y_test = split_data(features_x, labels_y)

    svc = LinearSVC(loss='hinge')
    svc.fit(X_train, y_train)

    labels_y_predicted = svc.predict(X_test)
    accuracy = get_prediction_accuracy(y_test, labels_y_predicted)
    print("Accuracy of " + accuracy)

    dist_pickle = {'svc': svc, 'x_scaler': X_scaler}
    pickle.dump(dist_pickle, open("svc_pickle.p", "wb"))


def process_image(frame, boxes_through_frames: BoxesDetectedWithCars):
    y_start_stop = [int(frame.shape[0] / 2), None]

    boxes_car_detected = []
    for scale in range(1, 4):
        boxes = find_cars(frame, svc, X_scaler, feature_extraction_parameters, scale=scale, y_start_stop=y_start_stop)
        for box in boxes:
            boxes_car_detected.append(box)

    boxes_through_frames.add_boxes(boxes_car_detected)

    # calculate heatmap
    heat = np.zeros_like(frame[:, :, 0]).astype(np.float)
    heat = add_heat(heat, boxes_through_frames.get_boxes())

    heat = apply_threshold(heat, 45)

    # Find final boxes from heatmap using label function
    labels = label(heat)
    draw_img = draw_labeled_bboxes(np.copy(frame), labels)
    return draw_img


boxesDetectedWithCars = BoxesDetectedWithCars()


def process_frame(frame):
    global boxesDetectedWithCars
    return process_image(frame, boxesDetectedWithCars)


def video_processing():
    white_output = 'extra_test_video_output.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process_frame)
    white_clip.write_videofile(white_output, audio=False)


# train()
svc, X_scaler = load_stored_classifier()

# Test classifier on a single test image

black_car = mpimg.imread('test_images/car_white.png')
frame_features = extract_features_for_single_image(feature_extraction_parameters, X_scaler, black_car)
black_car_label = svc.predict(frame_features)
print("predicted label for white_car: " + str(black_car_label))

# Process video

video_processing()
