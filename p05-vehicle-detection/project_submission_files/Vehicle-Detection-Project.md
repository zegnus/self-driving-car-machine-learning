# Vehicle Detection Project
The objective of this project is to detect vehicles in a video stream. In order to do that we will have to apply:

- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier.
- Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
- Create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
- Estimate a bounding box for vehicles detected.

This project and all its mentioned resources can be found in [**github**](https://github.com/zegnus/vehicle-detection)


# **Structure of the Project**
## **Source code**

I have generated three python files in the root directory of the project:

1. `vehicle_detection.py` is the main source code where high abstract functions are used. This file shows the intention of the algorithm in a high level abstract usage of functions.
2. `helper_functions.py` contains all the functions created for this project in order to proceed with feature abstractions, finding cars in images, etc.
3. `lesson_functions.py` contains mainly all the functions provided already by the course, with some minor improvements, expansions and modifications in order to better fit my needs.
4. `classes.py` contains two classes that will encapsulate major parameters and values used for training and for processing the bounding boxes through frames.
## **Assets**

Two main asset directories were created:

- `/dataset` contains the dataset that we are going to use for training purposes
- `/test_images` contains some training isolated frame images that I have used in order to calibrate and test the algorithms implemented.
## **Outputs**

As outputs, I have generated the following files in the root directory of the project:

- `svc_pickle.p` contains a trained `SVC` classifier and its associated `X_scaler` for normalisation.
- `project_video_output.mp4` contains the final output of the estimated detected vehicles based on the provided `project_video.mp4` file


# **Dataset**

The dataset that we are going to use is going to include images of [**cars**](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) (8792 images) and [**non-cars**](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) (8968 images) of `64x64` pixels. 

Loading all the dataset can be found in the method `extract_features_from_dataset` that also accepts a `sample_size` optional parameter in order to speed up testing and shuffleing is applied when the `sample_size` is used in order to provide enough diversity of samples:


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

![dataset](https://d2mxuefqeaa7sj.cloudfront.net/s_0DB92FEE8768AB6B798E30106F5F4039491EF793045E0887D975FC9C28974ACA_1494179386984_dataset.png)



# **Feature Extraction**

All the images are going to be guaranteed to be `RGB` and it will contain values from `[0 - 255]`. This is done by:


    if file.endswith("png"):
        image = cv2.imread(file)  # reads in BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = mpimg.imread(file)

I will also expand the dataset flipping horizontally all the images so that we include more cases to the classifier with the aim of improving its generalisation:

    cv2.flip(image, 1)

For extracting the features I have created a class `FeatureExtractionParameters` that encapsulates all the values that we can modify in order to create simpler and clearer methods.

I have proceed with a colour space transformation, histogram and HOG in order to provide with a set of features that the classifier will use:

## **Color transformation**

Through the function `convert_color` we can change the color space to different spaces. After trying different values I have decided to go for `YCrCb` color space transformation.

![color](https://d2mxuefqeaa7sj.cloudfront.net/s_0DB92FEE8768AB6B798E30106F5F4039491EF793045E0887D975FC9C28974ACA_1494182191903_color_space_transformation.png)


After this color space transformation, I stack all the three channels in a single list, rescaling the image to `16x16px` under the `bin_spatial` function:


    def bin_spatial(img, size=(16, 16)):
        color1 = cv2.resize(img[:, :, 0], size).ravel()
        color2 = cv2.resize(img[:, :, 1], size).ravel()
        color3 = cv2.resize(img[:, :, 2], size).ravel()
        return np.hstack((color1, color2, color3))
## **Histogram**

I calculate the histogram of all the three channels

![histogram](https://d2mxuefqeaa7sj.cloudfront.net/s_0DB92FEE8768AB6B798E30106F5F4039491EF793045E0887D975FC9C28974ACA_1494182195240_histogram.png)


And then I stack them in a single list:


    def color_hist(img, nbins=32, bins_range=(0, 256)):
        if np.max(img) <= 1.0:
            bins_range = (0, 1.0)
    
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    
        return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
## **HOG**

The HOG transformation can be tuned by modifying three main parameters. This parameters will define how many features we are going to get after the transformation.

I have tried several parameters between the margins being given by the examples in the course and I managed to get a good balance between accuracy and speed.

The parameters used for the HOG transformation are:


    orientations = 8
    pixels per cell = (8, 8) 
    cells per block = 2

![hog](https://d2mxuefqeaa7sj.cloudfront.net/s_0DB92FEE8768AB6B798E30106F5F4039491EF793045E0887D975FC9C28974ACA_1494182198362_hog_transformation.png)


We are going to get 8 different levels of gradients per every block; in which every block is composed by a square of two cells. The gradient is calculated between the cells defined. And finally, every cell contains a square of 8 pixels each.


# **Training and Prediction**

The training has been done with a linear Support Vector Machine classifier

    svc = LinearSVC(loss='hinge')
    svc.fit(X_train, y_train)

And I have generated a **training set** and a **test set** from the features generated previously.

From the features generated form the car dataset and from the not-car we are going to generate a normalised set of features and labels:

**1.** We stack the features in one list in order to have one single stream of features, the start is formed by the car features and the end is formed by the non-car features:


    X = np.vstack((car_features, notcar_features)).astype(np.float64)

**2.** We normalise the features through a [**StandardScaler**](http://scikit-learn.org/stable/modules/preprocessing.html). We are going to use this scaler afterwards for normalising any image-features that we want to classify:


    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

**3.** Finally we generate labels for the features. We are going to use **one** for the label representing cars, and **zero** for the label representing non-cars features:


    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

Once we have got all the features we are going to provide 20% of them as test features:


    def split_data(features_x, labels_y):
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        return train_test_split(features_x, labels_y, test_size=0.2, random_state=rand_state)

Now with a training and a test set we can measure the accuracy of our classifier. This accuracy has been used in order to tweak the parameters for the features extraction methods.


    labels_y_predicted = svc.predict(X_test)
    accuracy = get_prediction_accuracy(y_test, labels_y_predicted)
    
    def get_prediction_accuracy(correct_values, predicted_values):
        number_correct_matches = np.sum(correct_values == predicted_values)
        accuracy = (number_correct_matches / len(correct_values)) * 100
        return "{0:.2f}".format(accuracy)

The classifier achieves an accuracy of `99.34%` 


# **Sliding Windows Search**

The training has been done on individual car and non-car images of `64x64px`. Our next step is going to be to identify possible vehicles on a full frame containing a motorway screenshot with all its components including sky, lane, signals, etc.

In order to pass to the classifier portions of the frame we are going to create windows of several sizes, extract the same features as in the training set and then predict the label with our classifier.

One modification from the original feature extraction is that I am not going to extract the HOG from every single window but only from the bottom half of the entire frame.

The method `find_cars` defines all this process and also accepts a `scale` parameter. We are going to use this parameter in order to create different window sizes:


    boxes_car_detected = []
    for scale in range(1, 4):
        boxes = find_cars(frame, svc, X_scaler, feature_extraction_parameters, scale=scale, y_start_stop=y_start_stop)
        for box in boxes:
            boxes_car_detected.append(box)
    
    boxes_through_frames.add_boxes(boxes_car_detected)

There are two main parameters in this method that defines the size and the actual area of search:

- `scale` will start with a value of `1` for a `64x64px` window up to `256x256px` for a scale value of `4`. This range has been determined measuring the smallest and the biggest car that we want to be able to detect in the image.
- `cells_per_step` inside the `find_cars` method defines the distance between one window and the next one. After trial an error `2` has been a good balance between CPU load (number of windows) and accuracy.


![slidingwindows](https://d2mxuefqeaa7sj.cloudfront.net/s_0DB92FEE8768AB6B798E30106F5F4039491EF793045E0887D975FC9C28974ACA_1494187013630_sliding_windows.png)

# **Heatmap**

The classifier can generate false positives as seen in the previous section. In order to minimise them we are going to create a heat map and a filter on it.

The idea behind this heatmap is that false positives will occur less frequently than correct predictions. We are going to add up all the pixels inside the detected boxes and then apply a filter on the number of pixels in total inside those boxes.


    # calculate heatmap
    heat = np.zeros_like(frame[:, :, 0]).astype(np.float)
    heat = add_heat(heat, boxes_through_frames.get_boxes())

![heatmap](https://d2mxuefqeaa7sj.cloudfront.net/s_0DB92FEE8768AB6B798E30106F5F4039491EF793045E0887D975FC9C28974ACA_1494187454030_heat_map.png)


Now if we apply a threshold of 1 we are going to be able to remove the false positive from the detection:


    def apply_threshold(heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap


![threshold](https://d2mxuefqeaa7sj.cloudfront.net/s_0DB92FEE8768AB6B798E30106F5F4039491EF793045E0887D975FC9C28974ACA_1494187557569_threshold_heat_map.png)


Once we have boxes representing cars in the heatmap image we can generate back bounding boxes representing unique cars:


    labels = label(heat)
    draw_img = draw_labeled_bboxes(np.copy(frame), labels)
    
    def draw_labeled_bboxes(img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img


# **Video Stream**

We have seen that we can predict cars in a single frame. With a video stream we are going to continue to do the same operations but we are going to improve the threshold in the heatmap assuming that false positives will not hold for a long period of time.

I have created a helper class `BoxesDetectedWithCars` that will accumulate all the detected bounding boxes for `25` frames and then applying a heatmap threshold value of `45` I have managed to correctly detect the cars in the video stream in most of the situations.


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


# **Conclusions**

The classifier along with all the methods described previously are able to correctly approximate bounding boxes for the major cars that appear in the main `project_video.mp4` file.

The output of the project can be found in the root as `project_video_output.mp4` and also [**online**](https://youtu.be/KaJcl2gB744).


![conclusion](https://d2mxuefqeaa7sj.cloudfront.net/s_0DB92FEE8768AB6B798E30106F5F4039491EF793045E0887D975FC9C28974ACA_1494188541138_Screenshot+from+project_video_output.mp4.png)



# **Improvements**

**Classifier improvements**
We have seen that in order to use the SVM classifier we had to extract features from the dataset, and those features were sensitive to all the different parameters that we could choose from.

An alternative would have been to use a deep neural network with an architecture similar to LeNet in order to let the network to extract the significant features for our cars.

**Sliding Window Search improvements**
Providing windows for different sizes has been a non straight forward task and it has been also very high on CPU load due to the quantity of the windows created.

An improvement to the current implementation would be to use the same technique used in the Advanced Lane project. A car does not instantly appear in the screen at any position, instead there are a few very well defined places where that can happen. We can create our initial search windows at the sides of the road and at the center (end of the lanes).
Once the initial windows have detected a vehicle, we can expand the window and proceed with a new search as we can assume that in consecutive frames, a car would be found near the current detected window. This will speed up the search and track of vehicles in the screen, reduce false positives and decrease CPU load.

