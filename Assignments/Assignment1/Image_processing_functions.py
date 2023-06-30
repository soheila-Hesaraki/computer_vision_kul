import cv2
import os
import numpy as np


def remove_black_parts(frame, frame_with_black):
    # Get the dimensions of the black shape in the second image
    height, width, channels = frame_with_black.shape
    mask = cv2.inRange(frame_with_black, (0, 0, 0), (0, 0, 0))

    # Get the region of interest (ROI) in the first image
    roi = frame[0:height, 0:width]

    # Mask the ROI with the inverse of the mask created above
    masked_roi = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))

    # Combine the masked ROI with the second image
    result = cv2.bitwise_or(masked_roi, frame_with_black)
    return result


def sift_circular_detection(img, object_to_detect):
    # Convert both images to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_obj = cv2.cvtColor(object_to_detect, cv2.COLOR_BGR2GRAY)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    kp_img, des_img = sift.detectAndCompute(gray_img, None)
    kp_obj, des_obj = sift.detectAndCompute(gray_obj, None)

    # Initialize the FLANN matcher
    flann = cv2.FlannBasedMatcher()

    # Find the best match for the object to detect in the input image
    matches = flann.match(des_obj, des_img)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Get the coordinates of the keypoints that match between the object and input images
    obj_pts = np.float32([kp_obj[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    img_pts = np.float32([kp_img[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Calculate the homography matrix between the object and input images
    H, _ = cv2.findHomography(obj_pts, img_pts, cv2.RANSAC)

    # Get the bounding box of the object in the input image
    h, w = object_to_detect.shape[:2]
    obj_box = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    img_box = cv2.perspectiveTransform(obj_box, H)
    x, y, w, h = cv2.boundingRect(img_box)

    # Calculate the radius of the circle to draw
    radius = int(max(w, h) / 2)

    # Calculate the center of the circle to draw
    center = (int(x + w/2), int(y + h/2))

    return center, radius


def hugh(frame, step, MinDist, MinRadius, Param2, MaxRadius):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.medianBlur(gray_frame, 7)

    circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, 1, minDist=MinDist[step],
                               param1=35, param2=Param2[step], minRadius=MinRadius[step],
                               maxRadius=MaxRadius[step])
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    return frame


def grab_in_rgb(frame, lower_RGB, upper_RGB):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(frame, lower_RGB, upper_RGB)
    frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return frame


def grab_red_object_in_hsv(frame):
    # The color red is represented by two ranges of Hue values in the HSV color space.
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_mask = cv2.inRange(hsv, lower_red1, upper_red1)
    upper_mask = cv2.inRange(hsv, lower_red2, upper_red2)

    full_mask = lower_mask + upper_mask

    frame = cv2.bitwise_and(frame, frame, mask=full_mask)
    return frame


def sobel_horizontal_edge(frame):
    sobel_kernel_horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_sobel = cv2.filter2D(src=frame_gray, ddepth=-1, kernel=sobel_kernel_horizontal)

    # Threshold the image to create a binary mask of the edges
    _, frame_thresh = cv2.threshold(frame_sobel, 50, 255, cv2.THRESH_BINARY)

    # Dilate the edges to make them thicker
    kernel = np.ones((5, 5), np.uint8)
    frame_dilated = cv2.dilate(frame_thresh, kernel, iterations=1)

    # Create a yellow color mask for the edges
    yellow = np.array([0, 255, 255], dtype=np.uint8)
    frame_color = cv2.cvtColor(frame_dilated, cv2.COLOR_GRAY2BGR)
    frame_color[np.where((frame_color == [255, 255, 255]).all(axis=2))] = yellow

    return frame_color


def sobel_vertical_edges(frame):
    sobel_kernel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_sobel = cv2.filter2D(src=frame_gray, ddepth=-1, kernel=sobel_kernel_vertical)

    # Threshold the image to create a binary mask of the edges
    _, frame_thresh = cv2.threshold(frame_sobel, 50, 255, cv2.THRESH_BINARY)

    # Dilate the edges to make them thicker
    kernel = np.ones((5, 5), np.uint8)
    frame_dilated = cv2.dilate(frame_thresh, kernel, iterations=1)

    # Create a yellow color mask for the edges
    green = np.array([0, 255, 0], dtype=np.uint8)
    frame_color = cv2.cvtColor(frame_dilated, cv2.COLOR_GRAY2BGR)
    frame_color[np.where((frame_color == [255, 255, 255]).all(axis=2))] = green

    return frame_color


def sobel_combined_edges(frame):
    # Define Sobel kernels for both horizontal and vertical edge detection
    sobel_kernel_horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_kernel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Sobel filters to extract horizontal and vertical edges
    frame_sobel_horizontal = cv2.filter2D(src=frame_gray, ddepth=-1, kernel=sobel_kernel_horizontal)
    frame_sobel_vertical = cv2.filter2D(src=frame_gray, ddepth=-1, kernel=sobel_kernel_vertical)

    # Threshold the images to create binary masks of the edges
    _, frame_thresh_horizontal = cv2.threshold(frame_sobel_horizontal, 50, 255, cv2.THRESH_BINARY)
    _, frame_thresh_vertical = cv2.threshold(frame_sobel_vertical, 50, 255, cv2.THRESH_BINARY)

    # Dilate the edges to make them thicker
    kernel = np.ones((5, 5), np.uint8)
    frame_dilated_horizontal = cv2.dilate(frame_thresh_horizontal, kernel, iterations=1)
    frame_dilated_vertical = cv2.dilate(frame_thresh_vertical, kernel, iterations=1)

    # Create yellow and green color masks for the horizontal and vertical edges
    yellow = np.array([0, 255, 255], dtype=np.uint8)
    red = np.array([255, 0, 0], dtype=np.uint8)
    frame_color_horizontal = cv2.cvtColor(frame_dilated_horizontal, cv2.COLOR_GRAY2BGR)
    frame_color_horizontal[np.where((frame_color_horizontal == [255, 255, 255]).all(axis=2))] = yellow
    frame_color_vertical = cv2.cvtColor(frame_dilated_vertical, cv2.COLOR_GRAY2BGR)
    frame_color_vertical[np.where((frame_color_vertical == [255, 255, 255]).all(axis=2))] = red

    # Add the horizontal and vertical edge color masks together to create a combined image
    frame_color_combined = cv2.addWeighted(frame_color_horizontal, 1, frame_color_vertical, 1, 0)

    return frame_color_combined


def template_matching(frame, template):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

    # Get the location of the best match
    loc = cv2.minMaxLoc(res)
    return loc


def draw_boundry(frame, template_path, threshold):
    """This function draw a rectangular around the detected template that
     been detected by template matching"""
    template = cv2.imread(template_path)
    w, h = template.shape[1],  template.shape[0]
    loc = template_matching(frame, template)
    if loc[1] > threshold:
        # Draw a rectangle around the detected object
        top_left = loc[3]
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(frame, top_left, bottom_right, (225, 0, 0), 2)


def gray_scale_map(frame, template_path):
    # Convert to grayscale for reducing dimensionality of image matrix and increasing speed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_path)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # Match the template with the frame using cv.matchTemplate
    match = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF_NORMED)

    # Compute the inverse proportional likelihood
    likelihood = 1 - match

    # Normalize the likelihood to the range [0, 255]
    likelihood = cv2.normalize(likelihood, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the likelihood to an 8-bit integer
    gray_map = likelihood.astype('uint8')

    # Pad the output array to match the dimensions of the input frame
    height_diff = frame.shape[0] - gray_map.shape[0]
    width_diff = frame.shape[1] - gray_map.shape[1]
    gray_map = np.pad(gray_map, ((0, height_diff), (0, width_diff)), 'constant')
    gray_map = cv2.cvtColor(gray_map, cv2.COLOR_GRAY2BGR)

    return gray_map


def crop_image(image,x_start, y_start, x_end, y_end):
    cut_image = image[y_start:y_end, x_start:x_end]
    return cut_image


def circular_crop(image, object_to_detect):
    mask_image = np.zeros_like(image)
    center, radius = sift_circular_detection(image, object_to_detect)
    cv2.circle(mask_image, center, int(radius), (255, 255, 255), -1)
    masked_img = cv2.bitwise_and(image, mask_image)
    cropped_img = masked_img[int(center[1] - radius):int(center[1] + radius),
                             int(center[0] - radius):int(center[0] + radius)]
    return cropped_img


def circular_resize(image, circular_image, desired_radius):
    """Resize the circular image that is in the image to have the desired radius"""
    center, radius = sift_circular_detection(image, circular_image)
    scaling_factor = desired_radius / radius
    resized_circular_image = cv2.resize(circular_image, (0, 0), fx=scaling_factor, fy=scaling_factor)
    return resized_circular_image


def replace_circular_images(image, circular_image, new_circular_image):
    """ It replaces the circular image that is in the image with the new circular image"""
    center, radius = sift_circular_detection(image, circular_image)
    # # Calculate the top-left corner of the circular image in the image
    wheel_top_left = (int(center[0] - radius), int(center[1] - radius))

    # Replace the area in the image occupied by the circular with the new circular image
    image[wheel_top_left[1]:wheel_top_left[1] + new_circular_image.shape[0],
          wheel_top_left[0]:wheel_top_left[0] + new_circular_image.shape[1]] = new_circular_image
    return image


# brightness adjustment
def bright(img, beta_value ):
    img = cv2.convertScaleAbs(img, beta=beta_value)
    return img


def sharpen(img):
    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
    img_sharpen = cv2.filter2D(img, -1, kernel)
    return img_sharpen


def sepia(img):
    img_sepia = np.array(img, dtype=np.float64) # converting to float to prevent loss
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
    img_sepia[np.where(img_sepia > 255)] = 255 # normalizing values greater than 255 to 255
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    return img_sepia


def pencil_sketch_grey(img):
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    sk_gray = cv2.cvtColor(sk_gray, cv2.COLOR_GRAY2BGR)
    return  sk_gray


def pencil_sketch_col(img):
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    return  sk_color


def invert(img):
    inv = cv2.bitwise_not(img)
    return inv
