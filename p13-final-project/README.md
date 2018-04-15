# Programming a Real Self-Driving car

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59b6d115_final-project-ros-graph-v2/final-project-ros-graph-v2.png)

## Team

| Name                                           | Email                      | Responsibility                            |
| ---------------------------------------------- | -------------------------- | ----------------------------------------- |
| [Asad Zia](https://github.com/asadziach)       | asadzia@gmail.com          | Team Lead                                 |
| [Ferran Garriga](https://github.com/zegnus)    | zegnus@gmail.com           | Drive By Wire (Python)                    |
| [Leon Li](https://github.com/asimay)           | asimay_y@126.com           | Drive By Wire (C++)                       |
| [Mike Allen](https://github.com/mleonardallen) | mikeleonardallen@gmail.com | Waypoint Updater, Traffic Light (ROS)     |
| [Shahzad Raza](https://github.com/shazraz)     | raza.shahzad@gmail.com     | Traffic Light Detection (TensorFlow)      |

## Components
### **Basic Waypoint Updater**

First step was to implement a basic waypoint updator.

`/ros/src/waypoint_detector` This package contains the waypoint updater node: `waypoint_updater.py`. The purpose of this node is to update the target velocity property of each waypoint based on traffic light and obstacle detection data. This node will subscribe to the `/base_waypoints`, `/current_pose`, `/obstacle_waypoint`, and `/traffic_waypoint` topics, and publish a list of waypoints ahead of the car with target velocities to the `/final_waypoints` topic.

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/August/598d31bf_waypoint-updater-ros-graph/waypoint-updater-ros-graph.png)


**Steps**

The steps that we have followed in order to implement this module are the following:

1. Implement a loop in order to publish at 50hz our results
2. Calculate the closest waypoint from the car’s current location among the full list of waypoints injected to the code by the subscription
3. Calculate the next waypoint taking the direction’s car is facing into account from the closest waypoint and map’s coordinates
4. We will return a sublist of injected waypoints starting from the next calculated waypoint

**Implementation description**

We have implemented the solution in the file `waypoint_updater.py`, the major blocks implemented are the following:

**Main code loop**

We have implemented a main loop that cycles at 50hz and continuously publishes the calculated waypoints while we have some data available.

    def loop(self):
        rate = rospy.Rate(0.5)
        while not rospy.is_shutdown():
            if self.current_pose is None or self.base_waypoints is None:
                continue
    
            self.publish()
            rate.sleep()

**Publish**

From the [udacity project planning project](https://github.com/udacity/CarND-Path-Planning-Project/blob/59a4ffc9b56f896479a7e498087ab23f6db3f100/src/main.cpp#L64-L87), we will get the `next_waypoint` that’s closets to the direction of the car while `base_waypoints` is feeded into the code through a subscription `rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)`

The last step would be to publish the calculated waypoints to our publisher `self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)`


    def publish(self):
        """publish Lane message to /final_waypoints topic"""
    
        next_waypoint = self.next_waypoint()
        waypoints = self.base_waypoints.waypoints
        # shift waypoint indexes to start on next_waypoint so it's easy to grab LOOKAHEAD_WPS
        waypoints = waypoints[next_waypoint:] + waypoints[:next_waypoint]
        waypoints = waypoints[:LOOKAHEAD_WPS]
    
        lane = Lane()
        lane.waypoints = waypoints
        self.final_waypoints_pub.publish(lane)



### **DBW drive-by-wire**

`/ros/src/twist_controller` Carla is equipped with a drive-by-wire (dbw) system, meaning the throttle, brake, and steering have electronic control. This package contains the files that are responsible for control of the vehicle: the node `dbw_node.py` and the file `twist_controller.py`, along with a pid and lowpass filter that you can use in your implementation. The `dbw_node` subscribes to the `/current_velocity` topic along with the `/twist_cmd` topic to receive target linear and angular velocities. Additionally, this node will subscribe to `/vehicle/dbw_enabled`, which indicates if the car is under dbw or driver control. This node will publish throttle, brake, and steering commands to the `/vehicle/throttle_cmd`, `/vehicle/brake_cmd`, and `/vehicle/steering_cmd` topics.

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/August/598d32e7_dbw-node-ros-graph/dbw-node-ros-graph.png)

**Implementation description**
We implemented PID control with dynamic_reconfigure plugin first.

For PID control, use velocity from  `/twist_cmd`  and  `/current_velocity` as velocity CTE, use this value to handle speed_pid controller. we can use dynamic_reconfigure to adjust P,I,D’s value to get a good find tuned parameters.
usage:

    $ rosrun rqt_reconfigure rqt_reconfigure

to dynamic config the parameters of P,I,D. Following is screen snapshot.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_4BE7C326D61D0AB3964F4429284F019115EE07743CBB0A209367823E942FBEAC_1521636558771_image.png)


also, we used acceleration pid controller to control the throttle and brake.

![](https://d2mxuefqeaa7sj.cloudfront.net/s_4BE7C326D61D0AB3964F4429284F019115EE07743CBB0A209367823E942FBEAC_1521784481405_1.png)

![](https://d2mxuefqeaa7sj.cloudfront.net/s_4BE7C326D61D0AB3964F4429284F019115EE07743CBB0A209367823E942FBEAC_1521784505508_2.png)


Above is the PID testing screen snapshot, we need to test and tune a good PID value to make car run smoothly. Finally, we tuned the values as below:

    steer_kp = 3.0;
    ------
    speed_kp = 2.0;
    speed_ki = 1.0;
    speed_kd = 0;
    ------
    accel_kp = 0.4;
    accel_ki = 0.2;
    accel_kd = 0.2;

We also use lowpassfilter to filter the fuel variable, we assume it is 50% full of gas in simulator environment, but in real environment, we get the value from `/fuel_level_report` topic.
The DBW node subscribe the /current_velocity topic and /twist_cmd topic, we use the velocity from these two topic to do CTE estimation.
We get velocity CTE from:

    double vel_cte = twist_cmd_.twist.linear.x - cur_velocity_.twist.linear.x;

this drive our PID module to move.
there are also some condition check for twist_cmd velocity message, if too small, we reset PID error, in order to avoid accumulated error.

Then we use PID module to drive the vehicle’s throttle, brake. we also downloaded the udacity’s bagfile to do testing. snapshot as below:

![](https://d2mxuefqeaa7sj.cloudfront.net/s_4BE7C326D61D0AB3964F4429284F019115EE07743CBB0A209367823E942FBEAC_1521702232953_image.png)

After implementation the PID code, we found that the udacity configuration was missing paramters and config neededed for MPC. so we still focus on PID tuning and implementation.

We found the materials about MKZ’s control theory, so we did the code as what it said.
Following is the principle of control of MKZ:

![](https://d2mxuefqeaa7sj.cloudfront.net/s_4BE7C326D61D0AB3964F4429284F019115EE07743CBB0A209367823E942FBEAC_1521785256209_image.png)


The CTE come from speed command and speed measurement, according to related `/twist_cmd` and `/current_velocity`, we get cte from here, and then we use uses proportional control with gain `Kp`, this produced acceleration command, it is used to multiply m and r, to produce torque, which is `T=a*``*m**``r`, and combined losspass filter from speed measurement, we produced acceleration pid input, use PI controller to produce throttle pedal output.
During the whole process, we actually used all P,I,D method in it. The detailed info can be checked in code.


### **Traffic Light** **Detection & Classification**

`/ros/src/tl_detector` This package contains the traffic light detection node: `tl_detector.py`. This node takes in data from the `/image_color`, `/current_pose`, and `/base_waypoints` topics and publishes the locations to stop for red traffic lights to the `/traffic_waypoint` topic.

The `/current_pose` topic provides the vehicle's current position, and `/base_waypoints` provides a complete list of waypoints the car will be following.
You will build both a traffic light detection node and a traffic light classification node. Traffic light detection should take place within `tl_detector.py`, whereas traffic light classification should take place within `../tl_detector/light_classification_model/tl_classfier.py`.

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59b6d189_tl-detector-ros-graph/tl-detector-ros-graph.png)


**Steps**


1. Get closest light waypoint leveraging light data from the `/vehicle/traffic_lights` as well as the vehicles current location.  For closest waypoint, we converted some waypoint_updater.py code into a service so we could leverage the existing closest waypoint code in this module as well.
  
    msg = NextWaypointRequest(pose=pose, waypoints=waypoints)
    response = self.next_waypoint_proxy(msg)
  
2. Before classifying the image for light state, first check to see if the image is visible using a simple pinhole camera model to project the light waypoint onto the camera image.  Before we can do this however, we first need to convert the waypoint coordinates to coordinates with respect to the car or camera.  Conveniently, ROS provides a transformation module to perform these transformations.  
  
    transformed = self.transform_point(pose.header, point)

Once the light waypoint coordinates are transformed, we can then project onto the camera image.  If the projected point lies within bounding box of the image, then we say it is visible.

  
    # create camera info
    camera = PinholeCameraModel()
    camera.fromCameraInfo(self.camera_info)
    
    # project point onto image
    u, v = camera.project3dToPixel((x, y, z))
    
    # setup bounding box
    cx = camera.cx()
    cy = camera.cy()
    width = self.camera_info.width
    height = self.camera_info.height
    top = cy + height / 2
    bottom = cy - height / 2
    right = cx + width / 2
    left = cx - width / 2
    
    # light is visible if projected within the bounding box of the 2d image
    return left <= u and u <= right and bottom <= v and v <= top


3. If light is visible we send the current camera image from `/image_color`  topic, into the classifier by calling `self.light_classifier.get_classification(cv_image)`, which leverages a tensorflow model for classification.
4. The Tensorflow classification happens within `tl_classifier.py`.  Upon initialization of the module, we create a Tensorflow session and populate the default graph with a pre-trained protobuf file that contains the network definition and weights.
5. When `get_classification` is called, we pass in the given image to our Tensorflow model and request classification scores, labels, and bounding boxes.
6. From there, we keep the process fairly simple.  The classifier returns multiple bounding boxes, scores, and labels, but we just choose the label with the highest score that exceeds a threshold.  If no score exceeds the threshold, we return `TrafficLight.UNKNOWN`
7. The Tensorflow model uses slightly different class labels than those found in the Styx messages,  so before returning to `tl_detector` we first map the labels to the correct Styx counterpart.

```
    classifierToStyxLabelMap = {
        1: TrafficLight.GREEN,
        2: TrafficLight.RED,
        3: TrafficLight.YELLOW,
        4: TrafficLight.UNKNOWN
    }
    classfierLabel = detection_classes[0][0]
    return classifierToStyxLabelMap.get(classfierLabel)
```

8. Once we have the light state given by the classifier, we can finally publish the result to the `/traffic_waypoint` topic to allow the `waypoint_updater` to respond to traffic lights.

**TL Classification** **Implementation description**

Our implementation of the Traffic Light classification node uses TensorFlow’s Object Detection API. Since Carla’s environment uses Tensorflow v.1.3.0, we used a compatible version of the Object Detection API to evaluate various pre-trained models on the traffic light detection & classification task.

Our classification model is based on SSD Inception v2 trained on the MS COCO dataset available as part of the Tensorflow Model Zoo. This model was first fine-tuned on the Bosch small traffic light dataset. Two versions of the model were then created, one which was fine-tuned on simulator training images and the other on images from the Udacity test site. 

The training images were collected from the simulator and Slack. The training data available on slack was already annotated, however, the simulator training data required annotation. Faster R-CNN pre-trained on COCO was used to annotate the simulator images and the labels of the detected traffic lights were replaced with the color of the traffic light prior to saving the training data. The available training images were heavily biased towards red traffic lights so random translations were applied to augment the green and yellow traffic light images to balance the classes.


![](https://d2mxuefqeaa7sj.cloudfront.net/s_8480C1F0FD60A8575DA80FB264E7B89DECA0A84EE6E8E507922F5E3A0F1ACFB3_1523192464007_download.png)


The sections below provide details on how to use the Object Detection API to fine-tune a trained model on a custom dataset.

**Environment Preparation for TF Object Detection API**

1. Get the Tensorflow Object Detection API compatible with TF v1.3.0 by cloning the repository [here](https://github.com/tensorflow/models/tree/745a4481c5cbab8f7e7740802474cf4f34bbffee). Please note the specific commit version being used. 
2. Add the following directories to your PATH variable. Make sure this is done outside of any virtual environment so that the environment variables are inherited.
  1. `models/`
  2. `models/research/`
  3. `models/research/slim/`
**  *Linux users*: Append set these variables in your .bashrc file so this doesn’t need to be done for every shell.
  *Windows users*: Add these directories to your system environment variables 
3. Download the v3.4.0 of the Google protobuf binary for your OS from [here](https://github.com/google/protobuf/releases/tag/v3.4.0).
4. Extract the binary to `models/research/` and execute:
  1. *Linux users*: ``protoc --python_out=. object_detection/protos/*.proto``
  2. *Windows users*: `protoc` `--``python_out=. object_detection/protos/*.proto`
5. Verify that an equivalent python script has been created for each .proto file in `models/research/object_detection/protos/`
6. Setup the conda environment that contains versions of Tensorflow compatible with Carla by using the environment file [here](https://github.com/shazraz/tl-classifier/blob/master/capstone.yml). The environment name is `capstone`. *Note: This environment uses Python 3.5.2 for development whereas the environment on Carla uses Python 2.7.12.* 
7. Activate the `capstone` environment and execute the following from `models/research`:
  1. `python setup.py build`
  2. `python setup.py install`
8. Create a `configs/` folder in `models/research/object_detection/` to store custom model training configuration files.
9. Create a `custom_models/` folder with `trained_models` and `frozen_models` subfolders to hold models trained/fine-tuned on custom dataset.
10. Download a pre-trained model from the [Tensorflow Model Zoo](https://github.com/tensorflow/models/blob/745a4481c5cbab8f7e7740802474cf4f34bbffee/research/object_detection/g3doc/detection_model_zoo.md) compatible with your version of the Object Detection API. The COCO pre-trained models are a good starting point since they are already trained to detect a traffic light. 
11. Extract the model into `models/research/object_detection/` . The folder will contain a pre-trained model checkpoint, a frozen inference graph and the model graph.
12. The downloaded model can be evaluated on test image out of the box by using the `object_detection_tutorial` jupyter notebook included in `models/research/object_detection/` and the frozen inference graph included with the model. Additional test images can be put into the `object_detection/test_images/` folder. Make sure to update the jupyter notebook code block that reads in these images.

**Model Training**

The TF Object Detection API uses training data in the TFRecord format to train models. In addition, a corresponding label map file needs to be provided that replicates the labels used to create the TFRecord file. Control of the training parameters for various types of models is done via a model configuration file. Samples of the configuration files for various models are included in `object_detection/samples/configs/`

1. Create a TFRecord file for a dataset by using the conversion scripts [here](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-2-converting-dataset-to-tfrecord-47f24be9248d). Place this dataset in a new folder inside `object_detection/custom_models/trained_models/`. This folder will contain the trained checkpoints for a model trained on the created TFRecord.
2. Update the configuration file for the model to specify the number of classes, image dimensions, training data path, label map file path, learning rate, steps (epochs), etc. Sample configuration files for the Udacity dataset can be found [here](https://github.com/shazraz/tl-classifier/tree/master/config). Put the configuration file in your `object_detection/configs/` folder.
3. Create a label map file for the dataset and place it in `object_detection/data/`. Label maps for the Udacity and Bosch datasets can be found [here](https://github.com/shazraz/tl-classifier/tree/master/labels).
4. Navigate to `models/research/object_detection` and train the model by executing:

`python train.py --logtostderr --train_dir=custom_models/trained_models/<model_folder>/ --pipeline_config_path=configs/<model_config_file>.config`

5. Checkpoints will be created every 10 minutes in the folder containing the training dataset which can later be exported for inference.

**Exporting a model**

Once a trained checkpoint is available, a frozen graph can be exported for inference using the tools provided as part of the Object Detection API. To export a model, execute the following from `object_detection/`:
`python export_inference_graph.py --input_type image_tensor --pipeline_config_path configs/<model_config_file>.config --trained_checkpoint_prefix custom_models/trained_models/<model_folder>/model.ckpt-XXX --output_directory custom_models/frozen_models/<frozen_model_folder>`
Where XXX needs to be replaced by the step number at which the checkpoint was saved.

*Note: It is possible to export a frozen inference graph in a different version of TensorFlow than in which the model was originally trained. This can be used to export inference graphs from trained models that were added to model zoo in* [*later releases of the API*](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) *and used in older versions as long as there are no incompatibility issue.*

### **Full Way-point Updater**

**Implementation description**

The full waypoint updater starts with the partial waypoint updater, but now needs to be responsible for decelerating in response to red lights.  This module does not need to calculate jerk minimized trajectories (JMT), because the drive by wire (dbw) module already smoothly accelerates between waypoints.  For this reason, we simply need to give the car enough time to reach zero velocity through updating target velocities on the published `/final_waypoints`. Conversely, we do not need to give the car warning on green lights.  We can simply send full speed `base_waypoints`.

**Steps**

1. Receive red light information from the `/traffic_waypoint/` topic.

```
    rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
```

1. If red light, then send waypoints with velocities decelerated until reaching 0 at the given stopping waypoint.  We use a simple math square root function (as seen on walkthrough) to reduce the cars velocity as it gets closer to the given stopping waypoint.
  ```
    stop_wp_idx = self.traffic_waypoint.data
    stop_idx = max(stop_wp_idx - closest_idx - 2, 0) # Two waypoints from line so front of car stops at line
    dist = self.distance(waypoints, i, stop_idx)
    vel = math.sqrt(2 * self.max_decel * dist)
    if vel < 1.:
        vel = 0.
    
    p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
  ```

### Native Installation

Please use **one** of the two installation options, either native **or** docker installation.

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

**Logging and Debugging**

1. With styx running, open a new terminal tab
2. `docker ps` will give you list of running docker processes
3. run `docker exec -it <container id> bash` to enter same docker process that has styx running.
4. run `source devel/setup.sh`
5. For logging run `tail -f /root/.ros/log/latest/<log>`
6. For info run `rostopic info /<topic>` , etc

# Known Issues
- After cold boot on `tensorflow-gpu` setup, the first two runs are flaky, then it works ok. Logs indicate cuDNN failures.
- The `tensorflow-gpu` setup has limitation that TL process would crash if quality is set to `Fantastic` or resolution is higher than `800x600`.
- With CPU inference, the TL detection is slower, it take a while before car responds to TL transition. Resolution is not a problem, it works the same even on `2560x1920`. However if quality is set of `Fantastic`, second TL detection fails.

