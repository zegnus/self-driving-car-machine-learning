# Behavioural Cloning Project
The objective of the project is to drive a car in a simulator (or even in the real world) and then train a network that will copy your behaviour and apply it to the car itself in order to make it drive autonomously

**Project location**
All the code can be located in the following address:

https://github.com/zegnus/self-driving-car-nanodegree-behavioral-cloning

**The goals / steps of this project are the following:**

- Use a [**simulator**](https://github.com/udacity/self-driving-car-sim) in order to collect data of good driving behaviour
- Build a convolution neural network in [**Keras**](https://keras.io) that predicts the steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around the track one without leaving the road

**Project pre-requisites, resources and set-up**

- The project runs under **Python v3.5** and a set of libraries are [**required**](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/environment.yml)
- The full set-up can be found using the instructions of the [**Car Nanodegree Term 1 starter kit**](https://github.com/udacity/CarND-Term1-Starter-Kit)
- Once set-up, a closed environment will be supplied by [**miniconda**](https://conda.io/docs/install/quick.html) with all the dependencies correctly installed in isolation of your general environment
- For Python development I'm going to use [**Pycharm**](https://www.jetbrains.com/pycharm/download/#section=linux) and use the miniconda environment as the *Project Interpreter* in the project settings
![Interpreter](https://d2mxuefqeaa7sj.cloudfront.net/s_45B78DCDB3C3FD11E9779FD5428D8EAEF42B433C3AD4AB9548C142CCAFAD6CE4_1488647676686_project_interpreter.png)



# **Data-set**
## Data Collection

We want to ensure that we provide good driving data so that the network is able to generalise as much as possible. In order to do so, we will consider:

- The car should stay in the centre of the road as much as possible
- Driving counter-clockwise
- Flipping the images in order to augment the data
- Collecting data from the second track
## Format

The data-set is stored in the directory `resources/dataset`  and contains:

- A file  `driving_log.csv` 
  - Three **image** path file references for the three mounted cameras on the vehicle [centre, left and right]. 
  - **Steering angle**; values `[-1.2, 1.2]` 
  - Throttle; values  `[0, 1]` 
  - Break always  `0` 
  - Speed `[0, 30]` 
- A directory `IMG` that contains all the video frames captured by the simulator. The image shape is `160x320x3` 
## Data interpretation for a neural network training

In order to train a neural network, we need to define our **feature set** and our **label set**.
Our objective is that given a feature we are able to predict the correct label, is worth to recall that `Y = XW+b` 
For our dataset:

- features →all our images →`W` 
- labels → our steering values →`Y` 

We can extract the features and labels with the following code:

    import csv
    import cv2
    import numpy as np
    
    def load_csv(file_name):
        lines = []
        with open(file_name) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
        return lines
    
    def extract_features_labels(lines):
        features = []
        labels = []
        for line in lines:
            source_path = line[0]
            filename = source_path.split('/')[-1]
            current_path = 'resources/dataset/IMG/' + filename
            image = cv2.imread(current_path)
            features.append(image)
            labels.append(float(line[3]))
        return features, labels
    
    def parse_driving_log():
        lines = load_csv('resources/dataset/driving_log.csv')
        features, labels = extract_features_labels(lines)
        return np.array(features), np.array(labels)
    
    X_train, y_train = parse_driving_log()
## Composition of an image 

An image might contain the following visual components:

- The hood of the car at the bottom
- Environmental elements like sky, trees and other
- Lane and road

![Land_and_road](https://d2mxuefqeaa7sj.cloudfront.net/s_45B78DCDB3C3FD11E9779FD5428D8EAEF42B433C3AD4AB9548C142CCAFAD6CE4_1488729194251_center_2016_12_01_13_37_54_742.jpg)

# **Training**

The training is going to consist of several phases:

**1. Build a neural network model**

**2. Evaluate accuracy between the training set and the dataset**

  If the **accuracy is poor for both** we will could
  - Increase the number of epochs
  - Add more convolutions to the network

  If the **training set predicts well but the validation set predicts poorly** this is evidence of [**over-fitting**](https://en.wikipedia.org/wiki/Overfitting)
  - We can use [drop-out](https://medium.com/@vivek.yadav/why-dropouts-prevent-overfitting-in-deep-neural-networks-937e2543a701#.dox0ci345) or pooling layers
  - Reduce the number of convolutions or fully connected layers
  - Increase the data-set
  We can also print the accuracy processing the output from Keras:
    import matplotlib.pyplot as plt
    
    def print_loss(history_loss):
        plt.plot(history_loss.history['loss'])
        plt.plot(history_loss.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()
    
    history = model.fit(...)

**3. Test the model on the car simulator**

  Running the simulator in autonomous mode will require several steps:
  1. We will have the model saved in a file `model.h5` 
  2. Run the python file `drive.py model.h5` passing the model as a parameter
  3. Execute the simulator provided and click on the `autonomous` option
## **Dataset Processing**

We are provided with a pre-recorded data-set that contains 8036 samples in total and we will start using this set before recording more data.

### **Sub-sampling**
I have removed all recorded samples with the **throttle** value less than `0.25` . This prevents false steering values being learnt by the network when the car is actually not moving.

I have also tried to sub-sample by ranges of 90% - 50% all small steering angles (<0.05) from the original data-set due to the over-sampling of almost zero degrees caused by straight driving (see histogram in the following section Augmentation). 

Even though the histogram after this sub-sampling looked more normalised it had an undesired side effect. There is a section of the track that is straight but the pavement is of a different texture being a bridge. 

I deduced that this huge sub-sampling on straight angles reduced also the chances for the network to get samples from this particular part of the lane. Due to this side-effect I have decided to do not sub-sample in this particular way.

### **Normalisation**
Neural networks are very sensitive to big changes on the input data. In order to minimise that sensitivity and make the network more stable we should always normalise and pre-process the input data so that fits as best as possible for the network.

In Keras we can add [**lambda functions**](https://keras.io/layers/core/#lambda) that will allow us to inject any function that we want into the network execution. 

The code with lambdas for normalising the input will look like this:

    from keras.layers import Lambda
    
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
  - We normalise the values of the pixels to `[0, 1]` 
  - We shift the values towards the mean by `-0.5` all the pixels
  - We indicate the size of the input as this is our first layer (not needed for subsequent layers in Keras)

I have also restricted the steering values to units of `0.0004` . The simulator is giving us values from `[-25, 25]`  in units of `0.01` . Then the simulator recording scales down to the previously mentioned range of `1.2`  but the precision is too big giving us values as `0.05219137` . Restricting the steering angles to the nearest `0.0004` values gives us a more controlled and small range of values to work with without loosing too much precision.


    line[3] = round_nearest(float(line[3]), 0.0004)
    
    def round_nearest(x, a):
        return round(round(x / a) * a, -int(math.floor(math.log10(a))))
  

### **Image cropping**
In order to remove elements from the screen that does not affect the driving predictions we will crop the images 50px from the top and 20px from the bottom in order to remove environmental elements and the car hood.

    from keras.layers import Cropping2D
    
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

### **Augmentation - Adding more data**
A network will be as good as the data that we provide. Analysing the histogram of the data-set we can see that we have a huge amount of samples with almost no steering. This will make the network to be biased towards driving straight.

![Histogram](https://d2mxuefqeaa7sj.cloudfront.net/s_45B78DCDB3C3FD11E9779FD5428D8EAEF42B433C3AD4AB9548C142CCAFAD6CE4_1489838234825_original_dataset_histogram.png)


We should always aim to provide the same amount of samples for every desired output so that the network doesn't have a tendency towards any specific label

Also, providing a dataset restricted to a good driving on the first lap might not provide enough cases for the network to generalise. This will cause the car to turn to the left if we usually take left laps and to do not be able to stick to the centre when the car moves to the sides of the road as there are no samples with such scenario.

In order to help the network to properly create a generic behaviour we have to provide additional data.

We can extend the data-set manually recording from the simulator in order to provide missing samples. For example we can record the car positioned on the edge of the road and to recover from it moving towards the centre. This manual recording is extremely time consuming and we might be recording undesired cases. A cheap solution is to generate on-the-fly modified versions of the original data-set that would expand the number of cases.


#### 1. **Image mirroring**

In order to prevent the network to memorise the track we can augment the dataset by providing a mirroring of the original data:

    def mirror(feature, label):
        mirror_probability = np.random.random()
        if mirror_probability >= 0.5:
            feature = cv2.flip(feature, 1)
            label *= -1.0
    
        return feature, label


#### 2. **Left and right camera steering correction**

We are provided with screen-shots of the three cameras mounted in the car. We can use the left and the right cameras to simulate that the car is not on the centre of the image and so that it needs to steer to the centre. This data augmentation will help to keep the car in the centre of the line. Providing a correction angle too big and the car won’t be able to drive in a straight line, and providing a correction angle too low it won’t really help the correct the car from the sides to the centre of the lane.

Also I am **only correcting the further camera respect the steering angle**. This will help the car to steer stronger when it needs and also to avoid steer counter-wise when it doesn't need to. 

![extended_cameras](https://d2mxuefqeaa7sj.cloudfront.net/s_45B78DCDB3C3FD11E9779FD5428D8EAEF42B433C3AD4AB9548C142CCAFAD6CE4_1488913678677_Screenshot+from+2017-03-07+19-03-27.png)

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


#### 3. **Steering data-set augmentation with translations**

We can extend the number of images that needs steering by displacing horizontally and vertically the original image and apply a correction of every pixel translated:

    def get_random_translation(maximum_translation):
        return maximum_translation*np.random.uniform() - maximum_translation/2
    
    def translate(feature, label):
        horizontal_translation = get_random_translation(maximum_translation=100)
        vertical_translation = get_random_translation(maximum_translation=40)
    
        transformation_matrix = np.float32([[1, 0, horizontal_translation], [0, 1, vertical_translation]])
        output_size = (320, 160)
        feature = cv2.warpAffine(feature, transformation_matrix, output_size)
        # we add 0.004 steering angle for every translated pixel
        label += horizontal_translation * 0.004  
        return feature, label


#### 4. **Provide different brightness environments**

The network will be able to generalise better if we provide images with different levels of brightness. This will be a potential for preparing the model to be able to handle tracks with different light conditions (as in the second track)

    def modify_brightness(feature):
        feature = cv2.cvtColor(feature, cv2.COLOR_RGB2HSV)
        feature = np.array(feature, dtype=np.float64)
    
        random_bright = 0.25 + np.random.uniform()
    
        feature[:, :, 2] = feature[:, :, 2] * random_bright
        feature[:, :, 2][feature[:, :, 2] > 255] = 255
    
        feature = np.array(feature, dtype=np.uint8)
        return cv2.cvtColor(feature, cv2.COLOR_HSV2RGB)

In total we can provide four different augmentation methods

    def provide_augmented_dataset_from(line):
        feature, label = extract_feature_and_label_from(line)
        feature, label = mirror(feature, label)
        feature, label = translate(feature, label)
        feature = modify_brightness(feature)
        return feature, label

That will generate an extended data-set

![extended_data_set](https://d2mxuefqeaa7sj.cloudfront.net/s_45B78DCDB3C3FD11E9779FD5428D8EAEF42B433C3AD4AB9548C142CCAFAD6CE4_1489839373920_image_grid.png)


And the new histogram looks more distributed than the original data-set

![extended_data_set_histogram](https://d2mxuefqeaa7sj.cloudfront.net/s_45B78DCDB3C3FD11E9779FD5428D8EAEF42B433C3AD4AB9548C142CCAFAD6CE4_1489840126047_extended_dataset_histogram.png)


Manual samples were also recorded:


1. **Adding more laps**

The manual driving data-set for the first track is located under the folder `dataset\track_1_multiple_laps`  and contains **5011** samples


2. **Drive counter-clockwise**

The manual driving data-set counter-clockwise for the first track is located under the folder `dataset\track_1_multiple_laps_counter_clockwise` and contains **3505** samples


3. **Add different tracks**

The manual driving data-set for the second lap is located under the folder `dataset\track_2_multiple_laps`  with **5697** samples and the counter-clockwise is located under the folder `dataset\track_2_counter_clockwise` with **2207** samples

### **Generators**

Loading all the generated images all at once in memory could be really expensive. A better way is to load only a subset of the required images, process them and then remove them from memory and load the next batch. We can load just a **batch_size** at a time using the [**yield**](https://pythontips.com/2013/09/29/the-python-yield-keyword-explained/) python functionality.

Instead of loading into memory all the images in one array, with yield we can return a generator that loads a subset of the required images on-the-fly. The full code won’t be executed when the function is called. Instead the code will be executed until it reaches a yield, return those values and iterate again until there are no more values to return. So only the batch_size will be hold in memory at a time:

    def generate_samples(batch_size, lines):
        features = np.zeros((batch_size, 160, 320, 3))
        labels = np.zeros(batch_size)
    
        num_samples = len(lines)
    
        # we will generate as many images as batches
        for batch in range(batch_size):  
            line_index = np.random.randint(num_samples)
            line = lines[line_index]
    
            feature, label = provide_augmented_dataset_from(line)
            features[batch] = feature
            labels[batch] = label
    
        return features, labels
    
    def extract_features_labels_generator(lines):
        batch_size = 32
        
        lines = shuffle(lines)
    
        # Loop forever so the generator terminates when all samples are produced
        # it will start again for the next epoch
        while 1:  
            features, labels = generate_samples(batch_size, lines)
            yield features, labels
                
    def train_with_generator(model, train_lines, validation_lines, train_validation_ratio):
        generator_train = extract_features_labels_generator(train_lines)
        generator_validation = extract_features_labels_generator(validation_lines)
        history = model.fit_generator(
          generator_train,
          samples_per_epoch=desired_generated_training_samples,
          validation_data=generator_validation,
          nb_val_samples=desired_generated_validation_samples,
          verbose=1,
          nb_epoch=8,
        )

This generator wasn't good enough and it was failing to provide enough diversity of steerings. Having a look again to the generated histogram we can see the big concentration of values between `[-0.25, 0.25]` .

In order to solve this bias I have divided the steerings in three sections based on its histogram frequency.

Values with less than `0.25` degrees will only be added with a probability of `15%` 
Values between `[0.25 - 0.5]` degrees will be added with a probability of `75%` 
Values bigger than `0.5` degrees will always be included, probability of `100%` 

This will generate the following histogram providing more than double of sample for steering values bigger than `0.25` :

![Normalised_histogram](https://d2mxuefqeaa7sj.cloudfront.net/s_45B78DCDB3C3FD11E9779FD5428D8EAEF42B433C3AD4AB9548C142CCAFAD6CE4_1489937844259_extended_dataset_histogram.png)



## **Network architectures**

For our training we will always **split the training data-set** (all our features) into two groups, training and validation. The validation will be 20% of all our features. We do this in order to prevent the network to memorise all the dataset. Also the validation does not contain the augmented data-set.

The network will be trained by a total sample-set of **20480 samples per epoch** divided in batches of 32.

We will use the [**adam**](http://arxiv.org/pdf/1412.6980v8.pdf) optimiser for the calculation of our gradient and back-propagation.

In order to evaluation how well we are doing (training vs validation set) we will use the [**mean square error**](https://en.wikipedia.org/wiki/Mean_squared_error) as a regression function.

We will always **shuffle** our entire data-set every time that we execute the training (reference method `extract_features_labels_generator(...)` .


    model.compile(loss='mse', optimizer='adam')

### **Simple network**
As a starting point, we can create a very simple neural network:

    def simple_network_architecture(model):
        model.add(Flatten())
        model.add(Dense(1))
        return model

With image normalisation and cropping, the this simple network with 5 epochs the accuracy of this architecture is `loss: 1.1386 - val_loss: 1.6449` , also this model fails in the simulator autonomous test.

![simple_network](https://d2mxuefqeaa7sj.cloudfront.net/s_45B78DCDB3C3FD11E9779FD5428D8EAEF42B433C3AD4AB9548C142CCAFAD6CE4_1488833600362_figure_1.png)


This indicates that the model trains poorly for both training set and validation set, indicating that we probably need to **add more layers in our network**. Also the validation set loss is higher than the training set indicating that we are probably over-fitting, **augmenting the data set** should help. 

### **LeNet architecture**

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
        return model

With image normalisation and cropping, LeNet architecture on 5 epochs give us an accuracy of `loss: 0.0040 - val_loss: 0.0105` 

As we see the loss is far better than in the simple architecture but also this model fails in the simulator autonomous test.

![lenet](https://d2mxuefqeaa7sj.cloudfront.net/s_45B78DCDB3C3FD11E9779FD5428D8EAEF42B433C3AD4AB9548C142CCAFAD6CE4_1488910581184_figure_1.png)


Still, the validation set has a bigger loss than the training set indicating that we should probably augment the data.

### **LeNet with drop-out architecture**
The original LeNet was performing really bad in the simulator, but extending the LeNet with drop-out and with the full augmented data-set the simulator **was capable of driving by itself** successfully with just **three epochs**.

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

Comparing the loss with the original LeNet we can see that the training is got much better.

![lenet_with_dropout](https://d2mxuefqeaa7sj.cloudfront.net/s_45B78DCDB3C3FD11E9779FD5428D8EAEF42B433C3AD4AB9548C142CCAFAD6CE4_1489938791116_model_loss.png)


The successful produced model and a set of recorded autonomous driving can be found under the folder `resources/track_1_recording_lenet_with_dropout` 

Simulator in-car recording

https://www.youtube.com/watch?v=eUEnQ9-YzjY&


### **NVIDIA architecture**
NVIDIA published a [**paper**](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) presenting and end-to-end learning solution for their own set-up. I have taken the published model and add it into our training.

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

![nvidia](https://d2mxuefqeaa7sj.cloudfront.net/s_45B78DCDB3C3FD11E9779FD5428D8EAEF42B433C3AD4AB9548C142CCAFAD6CE4_1489938891071_model_loss.png)


This architecture has been able to produce a model that can drive the car in autonomous model in the first track for an unlimited amount of time. It’s interesting to notice that the loss is similar than the model predicted by LeNet with drop-out but the simulator was able to drive autonomously keeping the vehicle in the centre of the lane far better. 

Notice that the validation doesn't really mean a good prediction as the real validation occurs while driving in autonomous mode.

The successful produced model and a set of recorded autonomous driving can be found under the folder `resources/track_1_recording_nvidia` 

Simulator in-car recording

https://www.youtube.com/watch?v=APEwr1-eZyI&



# **Conclusion**

The **LeNet with dropout** and the **NVIDIA architecture** with the **pre-recorded dataset with procedural data augmentation** gave me the best results. The NVIDIA produced a more stable driving in the track number one.

Adding to the training the manually recorded data-set didn't increase the accuracy of the final model.

Even thought I have succeeded in developing a training model that can manage to drive a car in autonomous mode in the first track, it did not well enough in the second track. Probably extending the **data-set with more occlusion** could help to generalise more the model. Also guaranteeing the **same amount of samples for all desired steering** could also help. At the same time it is not possible in a real scenario to move the steering in a digital fashion; adding a **stabiliser on the predicted steering angles** frame after frame could also help to generate a more smooth driving. It would be interesting to drop frames similar to others in order to reduce data-set that does not add valuable information.

As next steps for the project I would probably start adding more intelligence on what’s captured by the cameras in the car and from the car sensors in order to build a coherence model of the world. As an example a car should stay in a lane and it wouldn't make sense to predict a steering angle that would move the car away from the lane with no reason; while testing different parameters and architectures the car freely moved away from the lane, this should never be allowed. Also the car should be able to analyse traffic signs and other cars behaviour. The model should be able to analyse other cars position and behaviour and produce a coherent response.

Designing a successful learning network is an extremely time consuming task due to the amount of hyper-parameters that we can tweak in the network itself, and the amount of different data-augmentation techniques that we can use.

Some generated data-set would actually have a negative impact on the network and even modifying the amount of epochs could make a model move from success to failure.

If we add the randomisation process on data-augmentation and model training we can easily see that this is a non-deterministic process.

Also the computational power required to perform many different trials and errors while modifying any parameter of our network is high. It’s highly recommended to use the parallel workforce power of **GPU**s in order to dramatically reduce training times.

For this project I have used the [**Amazon Elastic Computer Cloud**](https://aws.amazon.com/ec2/) (EC2) in order to execute the training in a machine with GPUs. It provided an invaluable resource while implementing this project. Without its services I wouldn't have been able to run all the training in my machine and finish the project on time or I would have had to consider buying a workstation with powerful GPUs. My workflow has been to design the network locally, upload the code to github and then use ssh for accessing the EC2 machine and pull from github.

A Dell XPS13 i7 takes `460s`  to run one epoch for the LeNet with dropout architecture, and the EC2 GPU takes `83s`  achieving an increase of performance of x5

![performance_cpu_vs_gpu](https://d2mxuefqeaa7sj.cloudfront.net/s_45B78DCDB3C3FD11E9779FD5428D8EAEF42B433C3AD4AB9548C142CCAFAD6CE4_1489858112397_Screenshot+from+2017-03-18+17-28-02.png)


Using EC2 with GPU is not free as there is a charge price for storage but it has been a cost-effective solution for my current set-up.



