# Advanced Lane Finding Project
The objective of the project is to detect with a high level of confidence the following parameters from a video of a car driving on a road:

- Detect the lane that the car is in
- Calculate the curvature of the lane
- Calculate the car horizontal offset respect to the center of the lane

We are provided with three videos `project_video.mp4` ,`challenge_video.mp4` and `harder_challenge_video.mp4` with a frame size of `1280x720` 

Also we have a set of pictures of a chessboard located in the folder `/camera_cal` and some frame images from the videos in the folder `test_images` . I have augmented this folder with more frames in order to test my algorithms in a more extended dataset.

The full project can be found on [**github**](https://github.com/zegnus/advanced-line-finding-project)

# **Overall solution description**

From the video feed we will extract every frame as an image and extract information from that frame. Also we will apply some logic cross-frames in order to fine-tune what would be an acceptable new prediction.

From a frame, we will:

- Compute the car’s camera optics distortion in order to correct it
- Convert the frame into a binary image with only the necessary information in order to proceed with the lane detection and clean the frame from noise and unnecessary information
- We will apply a perspective transformation to the image in order to provide a birds-eye view
- Locate the two lanes and fit a polynomial line that best describe the lanes
- Calculate the curvature of the lanes
- Calculate the offset of the car respect the center of the lanes
- Revert transformations and provide an overlay image on top of the original frame with augmented information (lanes detected, curvature and car offset from center)

As an example of a frame

![frame](https://d2mxuefqeaa7sj.cloudfront.net/s_9B79CE3FEBC771AF563FF279924CC9D369DCDFB68ED69F0B4302B464111E5AE0_1491647125636_lane-9.png)



# **Car optics distortion**

The camera mounted on the car distorts the stored image due to the optics. That is that the image is modified from the 3D world into a distorted 2D image. Usually the outer borders of the image will more distorted than the center.

In order to properly find properties about the 3D world we need to correct the optic’s distortions.

## **Chess Calibration**

We will images of a chessboard in order to find out the camera’s distortion parameters, the algorithm will find the corners of the chessboards and with a compilation of different images, [**cv2**](http://docs.opencv.org/2.4.8/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html) will be able to return back the parameters that define the camera’s distortion.

    def calibrate_camera():
        chess_shape = (9, 6)
        images = glob.glob("camera_cal/*")
        image_size = cv2.imread(images[0]).shape[0:2]
    
        image_points = []
        object_points = []
    
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0)
        objp = np.zeros((chess_shape[0] * chess_shape[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chess_shape[0], 0:chess_shape[1]].T.reshape(-1, 2)
    
        chessboard_corners = []
    
        for image_file in images:
            image = cv2.imread(image_file)
    
            #  To gray and we find the corners with no flags
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chess_shape, None)
    
            if ret == True:
                image_points.append(corners)
                object_points.append(objp)
    
                img = cv2.drawChessboardCorners(image, chess_shape, corners, ret)
                chessboard_corners.append(img)
    
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None)
    
        return mtx, dist

An example of the detected borders

![frame](https://d2mxuefqeaa7sj.cloudfront.net/s_9B79CE3FEBC771AF563FF279924CC9D369DCDFB68ED69F0B4302B464111E5AE0_1491646333352_chessboard_borders.jpg)


With the **transformation matrix** `mtx` and **distortion parameters** `dist` now we can correct the camera’s distortion

    def undistort(image, camera_matrix, distortion_parameters):
        return cv2.undistort(image, camera_matrix, distortion_parameters, None, camera_matrix)

The result of an image corrected

![frame](https://d2mxuefqeaa7sj.cloudfront.net/s_9B79CE3FEBC771AF563FF279924CC9D369DCDFB68ED69F0B4302B464111E5AE0_1491646457903_reverse_distortion.jpg.jpg)


We will use this function `undistort` in order to correct all the raw images from the video feed provided by the camera.
****
# **Image cropping**

In order to reduce unnecessary information from the image we will crop the frame.
I have analysed several frames from the three videos provided and end up the a set of parameters that can work for all of them.

    def crop(image):
        x1 = 150
        y1 = 480
        x2 = image.shape[1] - x1
        y2 = image.shape[0] - 45
        return image[y1:y2, x1:x2]

I also provide a method that will restore the cropping. This method will be used in the restoration end phase of the pipeline.

    def uncropp(destination, cropped_image):
        x = 150
        y = 480
        destination[y:y+cropped_image.shape[0], x:x+cropped_image.shape[1]] = cropped_image
        return destination

As an example of the cropping being applied to the dataset

![frame](https://d2mxuefqeaa7sj.cloudfront.net/s_9B79CE3FEBC771AF563FF279924CC9D369DCDFB68ED69F0B4302B464111E5AE0_1491647084470_cropped.jpg)

# **Perspective Transformation**

In order to properly calculate the curvature of the lane and other parameters, we will apply a perspective transformation to the frames. We will have as a result a birds-eye perspective from which we can easily process the image.

The algorithm will take four coordinates from the original image and we have to provide four coordinates of our desired perspective corrected final image. In order to accurately provide a birds-eye correctly transformed image we could physical take a picture from above the lanes but as we do not have such capability in this project, the four coordinates from the desired image has been a work of trial and error. In order to help me correct the transformation, I have researched the US standard highway lane dimensions and I have assumed that the lane is `3.2m` wide and that the white discontinued line is `1.5m` long

![frame](https://d2mxuefqeaa7sj.cloudfront.net/s_9B79CE3FEBC771AF563FF279924CC9D369DCDFB68ED69F0B4302B464111E5AE0_1491647661617_lane-1.png)

The algorithm will look like:

    def perspective_transformation(img):
        tl = (400, 0)
        tr = (586, 0)
        br = (908, img.shape[0] + 13)
        bl = (108, img.shape[0] + 13)
    
        src = np.float32([tl, tr, br, bl])
    
        left = (tl[0] + bl[0]) / 2
        right = (tr[0] + br[0]) / 2
    
        dst = np.float32(
            [
                (left, 0),  # top left
                (right, 0),  # top right
                (right, br[1]*4),  # bottom right
                (left, br[1]*4),  # bottom left
            ]
        )
    
        final_image_height = img.shape[0] * 4 + 30
        matrix_perspective = cv2.getPerspectiveTransform(src, dst)
    
        return cv2.warpPerspective(img, matrix_perspective, (img.shape[1], int(final_image_height)), flags=cv2.INTER_LINEAR)

I also provide a reverse perspective transformation that will be used in the final steps of the pipeline:

    def perspective_transformation_reverse(img):
        tl = (400, 0)
        tr = (586, 0)
        br = (918, 218)
        bl = (98, 218)
    
        src = np.float32([tl, tr, br, bl])
    
        left = 254
        right = 747
    
        dst = np.float32(
            [
                (left, 0),  # top left
                (right, 0),  # top right
                (right, 832),  # bottom right
                (left, 832),  # bottom left
            ]
        )
    
        final_image_height = (img.shape[0] - 30) / 4
        matrix_perspective = cv2.getPerspectiveTransform(dst, src)
    
        return cv2.warpPerspective(img, matrix_perspective, (img.shape[1], int(final_image_height)), flags=cv2.INTER_LINEAR)

The result of this perspective transformation applied to the dataset:

![frame](https://d2mxuefqeaa7sj.cloudfront.net/s_9B79CE3FEBC771AF563FF279924CC9D369DCDFB68ED69F0B4302B464111E5AE0_1491648465320_warp_lane-1.jpg)

# **Binary image thresholding**

There are many different ways to extract the pixels that are only related to the lanes in a road, here I have explored all channels on the RGB image color space and also all the channels on the HLS image color space.

The objective is to be invariant to brightness and shadows

I have also added filtering the image through [**Sobel**](https://en.wikipedia.org/wiki/Sobel_operator) filtering for gradient extraction (gradient, magnitude and direction)

    def sobel_threshold(image):
        # Choose a Sobel kernel size
        ksize = 3  # Choose a larger odd number to smooth gradient measurements
    
        # Apply each of the thresholding functions
        gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
        mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 120))
        dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.9, 1.60))
    
        combined = np.zeros_like(dir_binary, np.uint8)
        combined[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        return combined

An analysis of the dataset with different thresholding

![frame](https://d2mxuefqeaa7sj.cloudfront.net/s_9B79CE3FEBC771AF563FF279924CC9D369DCDFB68ED69F0B4302B464111E5AE0_1491648900020_color_threshold_lane-9.jpg)

With this analysis I was able to fine tune the threshold values for almost all filtering, but exploring the challenge videos that contains a big difference on pixel information I decided to apply a dynamic binary threshold to the red channel.

The idea is to split the image in to half. Every half will the calculate the maximum value and then apply a filter with a margin value. Then we will combine the two half in one image:

    def dynamic_threshold(image):
        half_image_position = image.shape[0]/2
        center_padding = 5
        lower_color_margin = 30
        max_left_value = np.max(image[:, :half_image_position-center_padding])
        max_right_value = np.max(image[:, half_image_position+center_padding:])
    
        left_half_image = image[:, :half_image_position]
        right_half_image = image[:, half_image_position:]
    
        binary_left_half = binary_threshold(left_half_image, (max_left_value - lower_color_margin, max_left_value))
        binary_right_half = binary_threshold(right_half_image, (max_right_value - lower_color_margin, max_right_value))
    
        binary = np.zeros_like(image)
        binary[:, :half_image_position] = binary_left_half
        binary[:, half_image_position:] = binary_right_half
    
        return binary

The binary resulting image is a combination of the dynamic threshold on the red channel and on top of it adding the full Sobel result. This adding will supplement the red channel filter in low light conditions:

    def binary_colorspace_thresholding(image):
        r = get_r_channel(image)
        r_binary = dynamic_threshold(r)
    
        s = get_s_channel(image)
        s_sobel = sobel_threshold(s)
    
        binary = np.zeros_like(r_binary)
        binary[(r_binary == 1) | (s_sobel == 1)] = 1
    
        return binary 


# **Binary lane polynomial fit**

In order to find the best polynomial fit for every lane we will process the image overall with the following steps:

**Start of the lanes**
Start from the bottom and find out though a histogram where we have the two lines. We count how many vertical white pixels are there in the bottom half of the image. Where we encounter a highest count, we will take the maximum position as the start of the lane.

**Polynomial fit**
We will create a window surrounding the initial pixels of the lines and counting the amount of lane pixels that fit in that window. We will then move the window vertically in order to find the continuation of the line from the previous bottom window. This will help the search for a coherent vertical line with the horizontal flexibility of the curves.

Once we have all pixels detected in windows, we’ll find the best polynomial fit for all those pixels:
`left_polynomial_fit = np.polyfit(left_lane_pixels_y, left_lane_pixels_x, 2)` 

Once we get the fit we will paint it in a mask image.

**Subsequent frames**
Using sliding windows for finding the lanes from scratch is expensive. We can take the previously polynomial fit and find again white pixels from the binary image that falls into a margin. We will do this for all valid subsequent frames with good established polynomial fits.

We will also compute the average of the polynomial coefficients for a span between `[7 - 19]` frames. This is due to the possibility of noise between immediate frames and we assume that not substantial changes can occur on the lines between immediate frames. A mean with a standard deviation could have been also used, but for our purposes the standard average was good enough.

For this averaging we are using a class

    class Line():
        def __init__(self):
            self.max_array_size = 10
            self.left_poly = []
            self.right_poly = []
            self.left_poly_av = []
            self.right_poly_av = []
            self.index = 0
    
        def average(self, left_polynomial, right_polynomial):
            if len(self.left_poly) < self.max_array_size:
                self.left_poly.append(left_polynomial)
                self.right_poly.append(right_polynomial)
      
            self.left_poly[self.index] = left_polynomial
            self.right_poly[self.index] = right_polynomial
            self.index += 1
            self.index %= self.max_array_size
            self.left_poly_av = np.average(self.left_poly, axis=0)
            self.right_poly_av = np.average(self.right_poly, axis=0)
            return self.left_poly_av, self.right_poly_av
    
        def get_left_stable_data(self):
            return self.left_poly_av
    
        def get_right_stable_data(self):
            return self.right_poly_av

The full method can be found on

    def find_initial_polynomial_fit(line: Line, image, polynomial_fit=()):
      # code

![frame](https://d2mxuefqeaa7sj.cloudfront.net/s_9B79CE3FEBC771AF563FF279924CC9D369DCDFB68ED69F0B4302B464111E5AE0_1491658573946_polymonial_fit_tests-small.jpg)

# **Lane curvature**

We are going to calculate the [**radius of the curvature**](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) of the lanes.

For a lane line that is close to vertical, you can fit a line using this formula: `f(y) = Ay^2 + By + C` 

- `A` gives you the curvature of the lane line
- `B` gives you the heading or direction that the line is pointing
- `C` gives you the position of the line based on how far away it is from the very left of an image (y = 0)

With a polynomial line we can easily calculate its curvature though the following function:

    left_curve = ((1 + (2 * left_polynomial_m[0] * np.max(left_y) + left_polynomial_m[1]) ** 2) ** 1.5) / np.absolute(2 * left_polynomial_m[0])

We would get the radius of the curvature but in pixel units. I have taken the bird-view of the frame and calculated the amount of pixels for the width of the lane and the length of the white discontinuous line. With this values I have been able to return the radius in meters.

    def get_angle(left_x, left_y, right_x, right_y):
        ym_per_pixel = 1.5 / 105.0
        xm_per_pixel = 3.2 / 282.0
    
        left_polynomial_m = np.polyfit(left_y * ym_per_pixel, left_x * xm_per_pixel, 2)
        right_polynomial_m = np.polyfit(right_y * ym_per_pixel, right_x * xm_per_pixel, 2)
    
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_polynomial_m[0] * np.max(left_y) * ym_per_pixel + left_polynomial_m[1]) ** 2) ** 1.5) \
                        / np.absolute(2 * left_polynomial_m[0])
        right_curverad = ((1 + (2 * right_polynomial_m[0] * np.max(right_y) * ym_per_pixel + right_polynomial_m[1]) ** 2) ** 1.5) \
                         / np.absolute(2 * right_polynomial_m[0])
    
        return left_curverad, right_curverad


# **Center of the lane and reverting the transformations**

**Center of the lane**
Once we got the polynomial fit, we will project the bottom of the image and calculate the middle point between the two polynomial fit results:

    def get_center_from_poly_fit(left_poly, right_poly, coordinate):
        bottom_left_line_polynomial = left_poly[0] * coordinate ** 2 + left_poly[1] * coordinate + left_poly[2]
        bottom_right_line_polynomial = right_poly[0] * coordinate ** 2 + right_poly[1] * coordinate + right_poly[2]
        return int((bottom_left_line_polynomial + bottom_right_line_polynomial) / 2)
    

With this pixel, now we will paint a dot in the masked image that contains the polynomial fit but with a different color.


    def draw_center_on(image, center):
        image[image.shape[0] - 1, center] = (255, 0, 0)
        return imag

**Revert transformation**
We will proceed to reverse the perspective transformation and to restore the cropping with the functions already provided. This will give us the lanes and a transformed dot (center of the lanes) projected into the original image.

Now we can scan the mask for the pixel that has the maximum amount of the color that we choose for painting the center, then taking half of the whole frame as the center of the car we will calculate the offset of the car respect the center of the lane.

    def get_center_from(image_with_drawn_center):
        return (np.argmax(image_with_drawn_center[image_with_drawn_center.shape[0] - 1], axis=0))[0]
    
    def get_center_difference(original_image, center_lines):
        return original_image.shape[1] / 2 - center_lines


# **Conclusion**

The algorithm and the pipeline designed **can successfully detect the lanes, curvature and car offset** in all the provided light and contrast situations on the `challenge-video` and in the `project-video` . You can find the output videos of the [**project-video**](https://youtu.be/--LzIl-mrBM) and [**challenge-video**](https://youtu.be/ij2aCRHYtnk)

![frame](https://d2mxuefqeaa7sj.cloudfront.net/s_9B79CE3FEBC771AF563FF279924CC9D369DCDFB68ED69F0B4302B464111E5AE0_1491732536248_project_video_output.png)

![frame](https://d2mxuefqeaa7sj.cloudfront.net/s_9B79CE3FEBC771AF563FF279924CC9D369DCDFB68ED69F0B4302B464111E5AE0_1491732544726_challenge_video_output.png)


We could further improve the project by:

- Doing an even deeper analysis on the Sovel capabilities
- Tweaking the thresholds and final composed image for binarisation
- Shorten the buffer frames in order to adapt faster to faster changes
- Dynamically buffer frames depending on the speed of the car
- Improve the polynomial fit with a mean standard deviation

