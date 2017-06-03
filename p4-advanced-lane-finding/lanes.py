import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from moviepy.editor import VideoFileClip

def print_images(images, titles, rows, columns, file_name, cmap=None, plot=[]):
    f, plots = plt.subplots(rows, columns, figsize=(24, 9))
    f.tight_layout()

    if rows == 1:
        for column in range(columns):
            if plot and plot[column]:
                plots[column].plot(images[column], cmap)
            else:
                plots[column].imshow(images[column], cmap)
            plots[column].imshow(images[column], cmap)
            if titles is not None:
                plots[column].set_title(titles[column])
    else:
        index = 0
        images_length = len(images)
        for row in range(rows):
            for column in range(columns):
                if index < images_length:
                    if plot and plot[index]:
                        plots[row][column].plot(images[index], cmap)
                    else:
                        plots[row][column].imshow(images[index], cmap)
                    if titles is not None:
                        plots[row][column].set_title(titles[index])
                    index += 1

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(file_name + ".jpg")
    plt.close()


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

    print_images(chessboard_corners, None, 2, 3, "output_images/chessboard_borders")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None)

    return mtx, dist


def undistort(image, camera_matrix, distortion_parameters):
    return cv2.undistort(image, camera_matrix, distortion_parameters, None, camera_matrix)


def test_undistortion():
    distorted_image = cv2.imread("camera_cal/calibration2.jpg")
    undistorted_image = undistort(distorted_image, camera_matrix, distortion_parameters)

    print_images([distorted_image, undistorted_image], None, 1, 2, "output_images/reverse_distortion.jpg")


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


def warp_all_test_images():
    images = glob.glob("test_images/*")
    for image_file in images:
        img = plt.imread(image_file)
        undistorted_image = undistort(img, camera_matrix, distortion_parameters)
        cropped_image = crop(undistorted_image)
        warp_image = perspective_transformation(cropped_image)
        print_images([cropped_image, warp_image], None, 1, 2, "output_images/warp_" + image_file.split("/")[1].split(".")[0])


def crop(image):
    x1 = 150
    y1 = 480
    x2 = image.shape[1] - x1
    y2 = image.shape[0] - 45
    return image[y1:y2, x1:x2]


def uncropp(destination, cropped_image):
    x = 150
    y = 480
    destination[y:y+cropped_image.shape[0], x:x+cropped_image.shape[1]] = cropped_image
    return destination


def uncropp_center_line(center_line):
    return center_line + 150


def crop_all_test_images():
    cropped_images = []
    images = glob.glob("test_images/*")
    for image_file in images:
        img = plt.imread(image_file)
        cropped_images.append(crop(img))

    print_images(cropped_images, None, 5, 4, "output_images/cropped")


def scale_if_need(image):
    max_value = np.max(image)
    if max_value <= 1.0:
        return cv2.convertScaleAbs(image, alpha=(255.0 / 1.0))
    return image


def get_r_channel(image):
    image = image[:, :, 0]
    return scale_if_need(image)


def get_g_channel(image):
    image = image[:, :, 1]
    return scale_if_need(image)


def get_b_channel(image):
    image = image[:, :, 2]
    return scale_if_need(image)


def get_h_channel(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:, :, 0]
    return scale_if_need(image)


def get_l_channel(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:, :, 1]
    return scale_if_need(image)


def get_s_channel(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:, :, 2]
    return scale_if_need(image)


def binary_threshold(image, threshold):
    binary = np.zeros_like(image)
    binary[(image > threshold[0]) & (image <= threshold[1])] = 1
    return binary


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    #  we calculate the absolute values
    abs_sobel = np.absolute(sobel)

    #  we transform it into an 8 bits image
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    #  we can create a binary threshold
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the binary image
    return sobel_binary


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    sobelx = np.absolute(sobelx)
    sobely = np.absolute(sobely)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    gradient_direction = np.arctan2(sobely, sobelx)

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(gradient_direction)

    # 6) Return this mask as your binary_output image
    binary_output[(gradient_direction >= thresh[0]) & (gradient_direction <= thresh[1])] = 1

    return binary_output


def histogram(image):
    return np.sum(image[image.shape[0]/2:, :], axis=0)


def one_line_color_quantity(image):
    return image[image.shape[0]/2]


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


def color_threshold_all_test_images():
    images = glob.glob("test_images/lane*")

    # images = ["test_images/test-1.png", "test_images/test-2.jpg", "test_images/test-3.jpg"]

    for image_file in images:
        img = plt.imread(image_file)
        undistorted_image = undistort(img, camera_matrix, distortion_parameters)
        cropped_image = crop(undistorted_image)
        warp_image = perspective_transformation(cropped_image)

        r = get_r_channel(warp_image)
        g = get_g_channel(warp_image)
        b = get_b_channel(warp_image)
        h = get_h_channel(warp_image)
        l = get_l_channel(warp_image)
        s = get_s_channel(warp_image)

        r_hist = one_line_color_quantity(r)
        g_hist = one_line_color_quantity(g)
        b_hist = one_line_color_quantity(b)
        h_hist = one_line_color_quantity(h)
        l_hist = one_line_color_quantity(l)
        s_hist = one_line_color_quantity(s)

        r_binary = dynamic_threshold(r)
        g_binary = binary_threshold(g, (170, 255))
        b_binary = binary_threshold(b, (200, 255))
        h_binary = binary_threshold(h, (10, 90))
        l_binary = binary_threshold(l, (140, 255))
        s_binary = binary_threshold(s, (50, 255))

        r_sobel = sobel_threshold(r)
        g_sobel = sobel_threshold(g)
        b_sobel = sobel_threshold(b)
        h_sobel = sobel_threshold(h)
        l_sobel = sobel_threshold(l)
        s_sobel = sobel_threshold(s)

        print_images(
            [r, g, b, h, l, s,
             r_hist, g_hist, b_hist, h_hist, l_hist, s_hist,
             r_binary, g_binary, b_binary, h_binary, l_binary, s_binary,
             r_sobel, g_sobel, b_sobel, h_sobel, l_sobel, s_sobel],
            ["r - channel", "g - channel", "b - channel", "h - channel", "l - channel", "s - channel",
             "r - color quantity", "g - color quantity", "b - color quantity", "h - color quantity", "l - color quantity", "s - color quantity",
             "r - binary", "g - binary", "b - binary", "h - binary", "l - binary", "s - binary",
             "r - sobel", "g - sobel", "b - sobel", "h - sobel", "l - sobel", "s - sobel"],
            4,
            6,
            "output_images/color_threshold_" + image_file.split("/")[-1].split(".")[0],
            cmap="gray",
            plot=[False, False, False, False, False, False,
                  True, True, True, True, True, True,
                  False, False, False, False, False, False,
                  False, False, False, False, False, False]
        )


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


def draw_rectangle_on(image, top_left, bottom_right):
    cv2.rectangle(
        image,
        top_left,
        bottom_right,
        (0, 255, 0),
        2
    )


def draw_rectangles(image, left_rect, right_rect):
    for i in range(len(left_rect)):
        draw_rectangle_on(image, left_rect[i][0], left_rect[i][1])
        draw_rectangle_on(image, right_rect[i][0], right_rect[i][1])
    return image


def draw_polynomial_fit(image, left_poly, right_poly):
    plot_y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_line_polynomial = left_poly[0] * plot_y ** 2 + left_poly[1] * plot_y + left_poly[2]
    right_line_polynomial = right_poly[0] * plot_y ** 2 + right_poly[1] * plot_y + right_poly[2]

    left = np.array([np.transpose(np.vstack([left_line_polynomial, plot_y]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_line_polynomial, plot_y])))])
    lane = np.hstack((left, right))

    cv2.fillPoly(image, np.int_([lane]), (0, 255, 0))

    return image


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


def find_polynomial_fit_sliding_windows(image, nonzero_x, nonzero_y):
    hist = histogram(image)

    # calculate left and right maximum
    midpoint = np.int(hist.shape[0] / 2)
    left_maximum = np.argmax(hist[:midpoint])
    right_maximum = np.argmax(hist[midpoint:]) + midpoint

    # we apply vertical window sliding
    number_vertical_windows = 9
    window_height = np.int(image.shape[0] / number_vertical_windows)

    # we initialise starting left and right positions
    left_current = left_maximum
    right_current = right_maximum
    # we set the width of the window to look for pixels
    window_margin = 25
    # we set the minimum number of pixels found in the window to force to recenter the window
    minimum_pixels_to_recenter = 50
    left_lane_pixels = []
    right_lane_pixels = []

    left_rectangle = []
    right_rectangle = []

    for window in range(number_vertical_windows):
        y_top = image.shape[0] - (window + 1) * window_height
        y_bottom = image.shape[0] - window * window_height
        left_x_left = left_current - window_margin
        left_x_right = left_current + window_margin
        right_x_left = right_current - window_margin
        right_x_right = right_current + window_margin
        # draw left and right windows
        left_rectangle.append(((left_x_left, y_top), (left_x_right, y_bottom)))
        right_rectangle.append(((right_x_left, y_top), (right_x_right, y_bottom)))

        # grab all pixels that are non zero inside the windows
        left_lane_pixels_in_window = (
            (nonzero_y >= y_top)
            & (nonzero_y < y_bottom)
            & (nonzero_x >= left_x_left)
            & (nonzero_x < left_x_right)
        ).nonzero()[0]

        right_lane_pixels_in_window = (
            (nonzero_y >= y_top)
            & (nonzero_y < y_bottom)
            & (nonzero_x >= right_x_left)
            & (nonzero_x < right_x_right)
        ).nonzero()[0]

        left_lane_pixels.append(left_lane_pixels_in_window)
        right_lane_pixels.append(right_lane_pixels_in_window)

        # recenter if necessary
        if len(left_lane_pixels_in_window) > minimum_pixels_to_recenter:
            left_current = np.int(np.mean(nonzero_x[left_lane_pixels_in_window]))

        if len(right_lane_pixels_in_window) > minimum_pixels_to_recenter:
            right_current = np.int(np.mean(nonzero_x[right_lane_pixels_in_window]))

    left_lane_pixels = np.concatenate(left_lane_pixels)
    right_lane_pixels = np.concatenate(right_lane_pixels)

    return (left_lane_pixels, right_lane_pixels), (left_rectangle, right_rectangle)


def find_polynomial_fit_from_previous_polynomial(nonzero_x, nonzero_y, polynomial_fit):
    margin = 20

    left_fit = polynomial_fit[0]
    right_fit = polynomial_fit[1]

    left_lane_pixels = ((nonzero_x > (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y + left_fit[2] - margin))
                        & (nonzero_x < (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y + left_fit[2] + margin)))
    right_lane_pixels = ((nonzero_x > (right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y + right_fit[2] - margin))
                         & (nonzero_x < (right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y + right_fit[2] + margin)))

    return left_lane_pixels, right_lane_pixels


class Line():
    def __init__(self):
        self.max_array_size = 10
        self.left_poly = []
        self.right_poly = []
        self.curvature_left = []
        self.curvature_right = []
        self.left_poly_av = []
        self.right_poly_av = []
        self.curvature_left_av = []
        self.curvature_right_av = []
        self.index = 0

    def average(self, left_polynomial, right_polynomial, curvature):
        if len(self.left_poly) < self.max_array_size:
            self.left_poly.append(left_polynomial)
            self.right_poly.append(right_polynomial)
            self.curvature_left.append(curvature[0])
            self.curvature_right.append(curvature[1])

        self.left_poly[self.index] = left_polynomial
        self.right_poly[self.index] = right_polynomial
        self.curvature_left[self.index] = curvature[0]
        self.curvature_right[self.index] = curvature[1]
        self.index += 1
        self.index %= self.max_array_size
        self.left_poly_av = np.average(self.left_poly, axis=0)
        self.right_poly_av = np.average(self.right_poly, axis=0)
        self.curvature_left_av = np.average(self.curvature_left)
        self.curvature_right_av = np.average(self.curvature_right)
        return self.left_poly_av, self.right_poly_av, (self.curvature_left_av, self.curvature_right_av)

    def get_left_stable_data(self):
        return self.left_poly_av

    def get_right_stable_data(self):
        return self.right_poly_av

    def get_latest_curvature(self):
        return self.curvature_left_av, self.curvature_right_av


def find_initial_polynomial_fit(line: Line, image, polynomial_fit=()):
    # we extract all no zero values
    nonzero = image.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    if polynomial_fit is ():
        lane_pixels, rectangle = find_polynomial_fit_sliding_windows(image, nonzero_x, nonzero_y)
    else:
        lane_pixels = find_polynomial_fit_from_previous_polynomial(nonzero_x, nonzero_y, polynomial_fit)
        rectangle = None

    left_lane_pixels = lane_pixels[0]
    right_lane_pixels = lane_pixels[1]

    # we get all left non zero pixels that are inside the windows
    left_x = nonzero_x[left_lane_pixels]
    left_y = nonzero_y[left_lane_pixels]
    right_x = nonzero_x[right_lane_pixels]
    right_y = nonzero_y[right_lane_pixels]

    # print("x:", len(right_x), "y:", len(right_y))

    minimum_pixels = 200
    left_polynomial = None
    right_polynomial = None

    if len(left_x) < minimum_pixels:
        left_polynomial = line.get_left_stable_data()

    if len(right_x) < minimum_pixels:
        right_polynomial = line.get_right_stable_data()

    if left_polynomial is None:
        left_polynomial = np.polyfit(left_y, left_x, 2)

    if right_polynomial is None:
        right_polynomial = np.polyfit(right_y, right_x, 2)

    if len(left_x) < minimum_pixels or len(right_x) < minimum_pixels:
        curvature = line.get_latest_curvature()
    else:
        curvature = get_angle(left_x, left_y, right_x, right_y)

    left_polynomial, right_polynomial, curvature = line.average(left_polynomial, right_polynomial, curvature)

    return True, (left_polynomial, right_polynomial), rectangle, curvature, line


def polynomial_fit_all_test_images():
    # images = ["test_images/lane-8.png", "test_images/lane-9.png"]
    images = glob.glob("test_images/*")

    f, plots = plt.subplots(len(images), 6, figsize=(24, len(images) * 3))
    f.tight_layout()

    for i in range(len(images)):
        img = plt.imread(images[i])
        undistorted_image = undistort(img, camera_matrix, distortion_parameters)
        cropped_image = crop(undistorted_image)
        image_color_thresholded = dynamic_threshold(get_r_channel(cropped_image))
        warp_image = perspective_transformation(cropped_image)
        binary = binary_colorspace_thresholding(warp_image)
        line = Line()
        ret, (left_poly, right_poly), (left_rect, right_rect), curvature, line = find_initial_polynomial_fit(line, binary)

        plots[i][0].imshow(img)
        plots[i][0].set_title("original")
        plots[i][1].imshow(cropped_image)
        plots[i][1].set_title("cropping")
        plots[i][2].imshow(warp_image)
        plots[i][2].set_title("perspective correction")
        plots[i][3].imshow(image_color_thresholded, cmap='gray')
        plots[i][3].set_title("color thresholding")
        plots[i][4].imshow(binary, cmap='gray')
        plots[i][4].set_title("binary thresholding")

        if ret == True:
            poly_image = np.copy(warp_image)
            for j in range(len(left_rect)):
                draw_rectangle_on(poly_image, left_rect[j][0], left_rect[j][1])
                draw_rectangle_on(poly_image, right_rect[j][0], right_rect[j][1])

            plot_y = np.linspace(0, poly_image.shape[0] - 1, poly_image.shape[0])
            left_line_polynomial = left_poly[0] * plot_y ** 2 + left_poly[1] * plot_y + left_poly[2]
            right_line_polynomial = right_poly[0] * plot_y ** 2 + right_poly[1] * plot_y + right_poly[2]

            plots[i][5].plot(left_line_polynomial, plot_y, color='yellow')
            plots[i][5].plot(right_line_polynomial, plot_y, color='yellow')
            plots[i][5].imshow(poly_image)
            plots[i][5].set_title("polynomial fit")
        else:
            print(images[i] + "not found")

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig("output_images/polymonial_fit_tests.jpg")
    plt.close()


def get_center_from_poly_fit(left_poly, right_poly, coordinate):
    bottom_left_line_polynomial = left_poly[0] * coordinate ** 2 + left_poly[1] * coordinate + left_poly[2]
    bottom_right_line_polynomial = right_poly[0] * coordinate ** 2 + right_poly[1] * coordinate + right_poly[2]
    return int((bottom_left_line_polynomial + bottom_right_line_polynomial) / 2)


def draw_center_on(image, center):
    image[image.shape[0] - 1, center] = (255, 0, 0)
    return image


def get_center_from(image_with_drawn_center):
    return (np.argmax(image_with_drawn_center[image_with_drawn_center.shape[0] - 1], axis=0))[0]


def get_center_difference(original_image, center_lines):
    return original_image.shape[1] / 2 - center_lines


def get_vehicle_center_offset_text(car_offset):
    car_offset_meters = car_offset * 3.2 / 750
    if car_offset < 0:
        side = "left"
    else:
        side = "right"

    car_offset_meters = np.abs(car_offset_meters)
    return "Vehicle is {0:.2f}m ".format(car_offset_meters) + side + " of center".format(car_offset_meters)


def process_image_pipeline_print(img, camera_matrix, distortion_parameters):
    f, plots = plt.subplots(2, 5, figsize=(24, 9))
    f.tight_layout()

    undistorted_image = undistort(img, camera_matrix, distortion_parameters)
    cropped_image = crop(undistorted_image)
    warp_image = perspective_transformation(cropped_image)
    image_color_thresholded = dynamic_threshold(warp_image)
    hist = histogram(image_color_thresholded)  # to plot
    binary = binary_colorspace_thresholding(warp_image)

    line = Line()

    ret, (left_poly, right_poly), rectangles, curvature, line = find_initial_polynomial_fit(line, binary)
    print("curvature left: {0:.2f}m ".format(curvature[0]) + "right: {0:.2f}m ".format(curvature[1]))

    mask = np.zeros_like(warp_image)
    center_lines = get_center_from_poly_fit(left_poly, right_poly, binary.shape[0])
    mask_with_center = draw_center_on(mask, center_lines)

    poly_fit = draw_polynomial_fit(mask_with_center, left_poly, right_poly)
    reverse_perspective = perspective_transformation_reverse(mask_with_center)

    center_lines = get_center_from(reverse_perspective)

    full_image_mask = np.zeros_like(img)
    full_image_uncropped = uncropp(full_image_mask, reverse_perspective)

    center_lines = uncropp_center_line(center_lines)
    car_offset = get_center_difference(img, center_lines)
    car_offset_string = "car center offset: " + str(car_offset)
    print(car_offset_string)

    text_vehicle_offset = get_vehicle_center_offset_text(car_offset)
    print(text_vehicle_offset)

    output = cv2.addWeighted(img, 1, full_image_uncropped, 0.2, 0.0)

    plots[0][0].imshow(undistorted_image)
    plots[0][0].set_title("original undistorted")

    plots[0][1].imshow(cropped_image)
    plots[0][1].set_title("cropped")

    plots[0][2].imshow(warp_image)
    plots[0][2].set_title("warp transformation")

    plots[0][3].imshow(image_color_thresholded, cmap='gray')
    plots[0][3].set_title("color threshold")

    plots[0][4].plot(hist)
    plots[0][4].set_title("histogram")

    plots[1][0].imshow(binary, cmap='gray')
    plots[1][0].set_title("binary threshold")

    plots[1][1].imshow(poly_fit)
    plots[1][1].set_title("polynomial fit")

    plots[1][2].imshow(reverse_perspective)
    plots[1][2].set_title("unwarp fit")

    plots[1][3].imshow(full_image_uncropped)
    plots[1][3].set_title("restore cropping fit")

    plots[1][4].imshow(output)
    plots[1][4].set_title("lanes detection")

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig("output_images/process_image_pipeline_print.jpg")
    plt.show()

    return output


def binary_colorspace_thresholding(image):
    r = get_r_channel(image)
    r_binary = dynamic_threshold(r)

    s = get_s_channel(image)
    s_sobel = sobel_threshold(s)

    binary = np.zeros_like(r_binary)
    binary[(r_binary == 1) | (s_sobel == 1)] = 1

    return binary


def process_image(line, img, polynomy_fit=()):
    undistorted_image = undistort(img, camera_matrix, distortion_parameters)
    cropped_image = crop(undistorted_image)
    warp_image = perspective_transformation(cropped_image)
    binary = binary_colorspace_thresholding(warp_image)
    ret, polynomy_fit, rectangles, curvature, line = find_initial_polynomial_fit(line, binary, polynomy_fit)

    center_lines = get_center_from_poly_fit(polynomy_fit[0], polynomy_fit[1], binary.shape[0])
    center_image = draw_center_on(np.zeros_like(warp_image), center_lines)
    reverse_perspective = perspective_transformation_reverse(center_image)
    center_lines = get_center_from(reverse_perspective)

    mask_with_center = np.zeros_like(warp_image)
    draw_polynomial_fit(mask_with_center, polynomy_fit[0], polynomy_fit[1])
    reverse_perspective = perspective_transformation_reverse(mask_with_center)

    full_image_mask = np.zeros_like(img)
    full_image_uncropped = uncropp(full_image_mask, reverse_perspective)
    center_lines = uncropp_center_line(center_lines)
    car_offset = get_center_difference(img, center_lines)

    output = cv2.addWeighted(img, 1, full_image_uncropped, 0.2, 0.0)

    text_vehicle_offset = get_vehicle_center_offset_text(car_offset)
    text_lanes_curvature = "Curvature left: {0:.2f}m, ".format(curvature[0]) + "right: {0:.2f}m ".format(curvature[1])

    draw_text(output, text_vehicle_offset, 50)
    draw_text(output, text_lanes_curvature, 80)

    return output, polynomy_fit, line


def draw_text(image, text, y_coordinate):
    cv2.putText(
        image,
        text,
        (600, y_coordinate),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.9,
        color=(200, 200, 200),
        thickness=2,
        lineType=cv2.LINE_AA
    )


polynomy_fit = ()
line = Line()


def process_frame(image):
    global polynomy_fit
    global line
    output, polynomy_fit, line = process_image(line, image, polynomy_fit)
    return output


def video_processing():
    white_output = 'project_video_output.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    # clip1 = VideoFileClip("challenge_video.mp4")
    white_clip = clip1.fl_image(process_frame)
    white_clip.write_videofile(white_output, audio=False)


camera_matrix, distortion_parameters = calibrate_camera()
#test_undistortion()
#crop_all_test_images()
#warp_all_test_images()
#color_threshold_all_test_images()
#polynomial_fit_all_test_images()

# img = plt.imread("test_images/lane-9.png")
# process_image_pipeline_print(img, camera_matrix, distortion_parameters)
# img = process_frame(img)
# plt.imshow(img)
# plt.show()

video_processing()
